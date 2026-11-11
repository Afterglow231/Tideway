// Native llama-server instrumentation and sequential output-token replay.
// Each sample calls the ORIGINAL sampler before replacing its returned token.
// The server consequently evaluates the forced token in its normal next decode
// step. No output sequence is passed to llama_decode as a prefill batch.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "llama.h"

struct common_sampler;
struct replay_control {
    uint64_t generation, count, consumed, forced, capacity, reserved[3];
};
static_assert(sizeof(replay_control) == 64);
struct token_event {
    uint64_t monotonic_ns;
    int32_t native_token, delivered_token, logit_index, forced;
    uint64_t generation, ordinal;
};
static_assert(sizeof(token_event) == 40);
struct batch_event {
    uint64_t start_ns, return_ns;
    int32_t tokens, first_position, last_position, return_code;
};
static_assert(sizeof(batch_event) == 32);

#ifdef REPLAY_PROBE_STUB
llama_token common_sampler_sample(common_sampler *, llama_context *, int index, bool) {
    return 7 + index;
}
extern "C" bool llama_vocab_is_eog(const llama_vocab *, llama_token token) { return token == 1; }
extern "C" int32_t llama_decode(llama_context *, llama_batch batch) {
    return batch.n_tokens ? 0 : -1;
}
#elif defined(REPLAY_PROBE_MAIN)
#include <cassert>
llama_token common_sampler_sample(common_sampler *, llama_context *, int, bool);
int main(int argc, char **) {
    int fd = open(std::getenv("TIDEWAY_REPLAY_CONTROL"), O_RDWR);
    assert(fd >= 0);
    void *mapping = mmap(nullptr, 80, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(mapping != MAP_FAILED);
    auto *control = static_cast<replay_control *>(mapping);
    auto *ids = reinterpret_cast<int32_t *>(control + 1);
    ids[0] = 11;
    ids[1] = 12;
    ids[2] = 13;
    control->count = 3;
    control->forced = 1;
    __atomic_store_n(&control->generation, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < 3; ++i)
        assert(common_sampler_sample(nullptr, nullptr, i, false) == 11 + i);
    if (argc > 1)
        common_sampler_sample(nullptr, nullptr, 3, false); // Must fail closed.
    assert(control->consumed == 3);
    control->count = 2;
    control->forced = 0;
    __atomic_store_n(&control->generation, 2, __ATOMIC_RELEASE);
    for (int i = 0; i < 2; ++i)
        assert(common_sampler_sample(nullptr, nullptr, i, false) == 7 + i);
    assert(control->consumed == 2);
    assert(!llama_vocab_is_eog(nullptr, 1));
    llama_pos positions[] = {100, 101};
    llama_batch batch{};
    batch.n_tokens = 2;
    batch.pos = positions;
    assert(llama_decode(nullptr, batch) == 0);
    std::puts("NATIVE_SAMPLER_EXECUTED_FORCED_SEQUENCE_AND_RESET_PASS");
}
#else
static int events_fd = -1, batches_fd = -1;
static replay_control *control = nullptr;
static int32_t *forced_ids = nullptr;
static uint64_t active_generation = 0, ordinal = 0, mapped_capacity = 0;

static uint64_t now_ns() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return uint64_t(now.tv_sec) * 1000000000ull + uint64_t(now.tv_nsec);
}
static void append_event(int fd, const void *value, size_t size) {
    const char *data = static_cast<const char *>(value);
    while (size) {
        ssize_t n = write(fd, data, size);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            _exit(103);
        data += n;
        size -= n;
    }
}
__attribute__((constructor)) static void initialize_replay() {
    const char *events = std::getenv("TIDEWAY_REPLAY_TOKEN_EVENTS");
    const char *batches = std::getenv("TIDEWAY_REPLAY_BATCH_EVENTS");
    const char *path = std::getenv("TIDEWAY_REPLAY_CONTROL");
    if (!events || !batches || !path)
        _exit(100);
    events_fd = open(events, O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
    batches_fd = open(batches, O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0600);
    if (events_fd < 0 || batches_fd < 0)
        _exit(101);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) || st.st_size < int64_t(sizeof(replay_control)))
        _exit(106);
    void *mapping = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED)
        _exit(106);
    control = static_cast<replay_control *>(mapping);
    forced_ids = reinterpret_cast<int32_t *>(control + 1);
    mapped_capacity = (st.st_size - sizeof(replay_control)) / sizeof(int32_t);
    if (control->capacity != mapped_capacity)
        _exit(106);
}

llama_token common_sampler_sample(common_sampler *sampler, llama_context *context, int index,
                                  bool grammar_first) {
    using original_t = llama_token (*)(common_sampler *, llama_context *, int, bool);
    static original_t original = reinterpret_cast<original_t>(
        dlsym(RTLD_NEXT, "_Z21common_sampler_sampleP14common_samplerP13llama_contextib"));
    if (!original)
        _exit(102);
    // Runs the unmodified forward-dependent native sampler even in replay mode.
    const llama_token native = original(sampler, context, index, grammar_first);
    const uint64_t generation = __atomic_load_n(&control->generation, __ATOMIC_ACQUIRE);
    if (!generation)
        _exit(107);
    if (generation != active_generation) {
        active_generation = generation;
        ordinal = 0;
    }
    if (control->count > mapped_capacity || ordinal >= control->count)
        _exit(107);
    const bool forced = control->forced != 0;
    const llama_token delivered = forced ? forced_ids[ordinal] : native;
    if (delivered < 0)
        _exit(108);
    token_event event{now_ns(), native, delivered, index, int32_t(forced), generation, ordinal};
    append_event(events_fd, &event, sizeof(event));
    ++ordinal;
    __atomic_store_n(&control->consumed, ordinal, __ATOMIC_RELEASE);
    return delivered;
}
extern "C" int32_t llama_decode(llama_context *context, llama_batch batch) {
    using original_t = int32_t (*)(llama_context *, llama_batch);
    static original_t original = reinterpret_cast<original_t>(dlsym(RTLD_NEXT, "llama_decode"));
    if (!original)
        _exit(105);
    batch_event event{now_ns(),
                      0,
                      batch.n_tokens,
                      batch.pos && batch.n_tokens ? batch.pos[0] : -1,
                      batch.pos && batch.n_tokens ? batch.pos[batch.n_tokens - 1] : -1,
                      0};
    int32_t result = original(context, batch);
    event.return_ns = now_ns();
    event.return_code = result;
    append_event(batches_fd, &event, sizeof(event));
    return result;
}
extern "C" bool llama_vocab_is_eog(const llama_vocab *, llama_token) {
    // Disable stopping, without masking EOG logits or changing native sampling.
    return false;
}
#endif
