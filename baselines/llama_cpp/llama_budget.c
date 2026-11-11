/* Process-RSS budget governor for read-only GGUF mappings.
 * Window recency uses sampled residency growth, not exact page-access LRU.
 * This simulates reclaim; it does not impose a kernel or whole-host limit.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define MAX_WINDOWS 4096
#define SAMPLE_PAGES 32
#define SAMPLE_POSITIONS 16

struct window {
    uintptr_t lo, hi;
    off_t file_offset;
    uint64_t recency;
    unsigned resident_sample;
};
static struct window windows[MAX_WINDOWS];
static size_t count;
static pthread_mutex_t map_lock = PTHREAD_MUTEX_INITIALIZER;
static dev_t target_dev;
static ino_t target_ino;
static int target_fd = -1, active;
static size_t budget, margin, window_bytes, page_bytes;
static char log_path[1024], control_path[1024], ack_path[1024];
static long period_ms = 10;
static uint64_t ticks, total_evictions, total_reclaimed, peak_rss, purge_epoch;
static uint64_t governor_cpu_ns;
static int (*real_posix_madvise)(void *, size_t, int);

static uint64_t time_ns(clockid_t which) {
    struct timespec t;
    clock_gettime(which, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

static size_t read_rss(void) {
    char data[8192];
    int fd = open("/proc/self/status", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return SIZE_MAX;
    ssize_t n = read(fd, data, sizeof(data) - 1);
    close(fd);
    if (n <= 0)
        return SIZE_MAX;
    data[n] = 0;
    const char *p = strstr(data, "VmRSS:");
    if (!p)
        return SIZE_MAX;
    return strtoull(p + 6, NULL, 10) * 1024;
}

static int target_file(int fd) {
    struct stat s;
    return fd >= 0 && fstat(fd, &s) == 0 && s.st_dev == target_dev && s.st_ino == target_ino;
}

static void remove_range(uintptr_t lo, uintptr_t hi) {
    for (size_t i = 0; i < count;) {
        struct window old = windows[i];
        if (hi <= old.lo || lo >= old.hi) {
            i++;
            continue;
        }
        windows[i] = windows[--count];
        if (lo > old.lo) {
            if (count >= MAX_WINDOWS)
                _exit(91);
            struct window left = old;
            left.hi = lo;
            left.resident_sample = 0;
            windows[count++] = left;
        }
        if (hi < old.hi) {
            if (count >= MAX_WINDOWS)
                _exit(91);
            struct window right = old;
            right.file_offset += hi - old.lo;
            right.lo = hi;
            right.resident_sample = 0;
            windows[count++] = right;
        }
    }
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    int tracked = active && target_file(fd);
    if (tracked && (prot & PROT_WRITE))
        _exit(99);
    int locked = active && (tracked || (flags & MAP_FIXED));
    if (locked)
        pthread_mutex_lock(&map_lock);
    if (tracked)
        flags &= ~MAP_POPULATE;
    void *result = (void *)syscall(SYS_mmap, addr, len, prot, flags, fd, off);
    if (result != MAP_FAILED && locked) {
        uintptr_t lo = (uintptr_t)result;
        uintptr_t hi = lo + ((len + page_bytes - 1) / page_bytes) * page_bytes;
        remove_range(lo, hi);
        if (tracked) {
            for (uintptr_t a = lo; a < hi; a += window_bytes) {
                if (count >= MAX_WINDOWS)
                    _exit(91);
                uintptr_t end = a + window_bytes < hi ? a + window_bytes : hi;
                windows[count++] = (struct window){a, end, off + (off_t)(a - lo), ticks, 0};
            }
        }
    }
    if (locked)
        pthread_mutex_unlock(&map_lock);
    return result;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t off) {
    return mmap(addr, len, prot, flags, fd, (off_t)off);
}

int munmap(void *addr, size_t len) {
    if (active)
        pthread_mutex_lock(&map_lock);
    int rc = (int)syscall(SYS_munmap, addr, len);
    if (active && rc == 0)
        remove_range((uintptr_t)addr,
                     (uintptr_t)addr + ((len + page_bytes - 1) / page_bytes) * page_bytes);
    if (active)
        pthread_mutex_unlock(&map_lock);
    return rc;
}

int posix_madvise(void *addr, size_t len, int advice) {
    if (active && advice == POSIX_MADV_WILLNEED) {
        uintptr_t lo = (uintptr_t)addr, hi = lo + len;
        int overlaps = 0;
        pthread_mutex_lock(&map_lock);
        for (size_t i = 0; i < count; i++)
            if (lo < windows[i].hi && hi > windows[i].lo) {
                overlaps = 1;
                break;
            }
        pthread_mutex_unlock(&map_lock);
        if (overlaps)
            return 0;
    }
    if (!real_posix_madvise)
        real_posix_madvise = dlsym(RTLD_NEXT, "posix_madvise");
    if (!real_posix_madvise)
        return ENOSYS;
    return real_posix_madvise(addr, len, advice);
}

static void affinity(int pressure) {
    const char *configured = getenv(pressure ? "TIDEWAY_LLAMA_CPUS" : "TIDEWAY_LLAMA_MONITOR_CPU");
    if (!configured || !*configured)
        return;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    const char *at = configured;
    while (*at) {
        char *end;
        long first = strtol(at, &end, 10);
        if (end == at || first < 0 || first >= CPU_SETSIZE)
            _exit(92);
        long last = first;
        at = end;
        if (*at == '-') {
            const char *begin = ++at;
            last = strtol(at, &end, 10);
            if (end == begin || last < first || last >= CPU_SETSIZE)
                _exit(92);
            at = end;
        }
        for (long cpu = first; cpu <= last; cpu++)
            CPU_SET(cpu, &cpus);
        if (*at && *at != ',')
            _exit(92);
        if (*at == ',') {
            at++;
            if (!*at)
                _exit(92);
        }
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus))
        _exit(92);
}

static void probe_windows(void) {
    unsigned char resident[SAMPLE_PAGES];
    ticks++;
    for (size_t i = 0; i < count; i++) {
        struct window *w = &windows[i];
        size_t pages = (w->hi - w->lo) / page_bytes;
        size_t block = pages < SAMPLE_PAGES ? pages : SAMPLE_PAGES;
        size_t stride = pages / SAMPLE_POSITIONS;
        if (stride < block)
            stride = block;
        unsigned sum = 0;
        for (size_t p = 0; block && p + block <= pages; p += stride) {
            if (mincore((void *)(w->lo + p * page_bytes), block * page_bytes, resident))
                _exit(93);
            for (size_t n = 0; n < block; n++)
                sum += resident[n] & 1;
        }
        if (sum > w->resident_sample)
            w->recency = ticks;
        w->resident_sample = sum;
    }
}

static int release_window(struct window *w) {
    if (madvise((void *)w->lo, w->hi - w->lo, MADV_DONTNEED))
        return errno;
    int rc = posix_fadvise(target_fd, w->file_offset, w->hi - w->lo, POSIX_FADV_DONTNEED);
    if (!rc) {
        w->resident_sample = 0;
        w->recency = ticks;
        total_evictions++;
    }
    return rc;
}

static void acknowledge(uint64_t epoch, size_t rss) {
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", ack_path);
    FILE *f = fopen(tmp, "w");
    if (!f)
        _exit(94);
    fprintf(f, "{\"epoch\":%llu,\"rss_bytes\":%zu,\"pid\":%d}\n", (unsigned long long)epoch, rss,
            getpid());
    fclose(f);
    if (rename(tmp, ack_path))
        _exit(94);
}

static void *govern(void *unused) {
    (void)unused;
    affinity(0);
    FILE *log = fopen(log_path, "a");
    if (!log)
        _exit(94);
    setvbuf(log, NULL, _IOLBF, 0);
    double last_log = 0;
    for (;;) {
        struct timespec delay = {period_ms / 1000, (period_ms % 1000) * 1000000L};
        nanosleep(&delay, NULL);
        size_t rss = read_rss();
        if (rss == SIZE_MAX)
            _exit(95);
        if (rss > peak_rss)
            peak_rss = rss;
        uint64_t requested = purge_epoch;
        FILE *control = fopen(control_path, "r");
        if (control) {
            unsigned long long b = 0, e = 0;
            if (fscanf(control, "%llu %llu", &b, &e) == 2 && b > margin) {
                budget = b;
                requested = e;
            }
            fclose(control);
        }
        uint64_t before_count = total_evictions;
        int error = 0;
        if (requested != purge_epoch || rss > budget - margin) {
            affinity(1);
            uint64_t cpu_start = time_ns(CLOCK_THREAD_CPUTIME_ID);
            pthread_mutex_lock(&map_lock);
            if (requested != purge_epoch) {
                for (size_t i = 0; i < count && !error; i++)
                    error = release_window(&windows[i]);
                if (!error)
                    error = posix_fadvise(target_fd, 0, 0, POSIX_FADV_DONTNEED);
                purge_epoch = requested;
            } else {
                probe_windows();
                size_t goal = budget - margin - margin / 2;
                for (size_t tried = 0; tried < count && rss > goal; tried++) {
                    size_t victim = SIZE_MAX;
                    for (size_t i = 0; i < count; i++) {
                        if (!windows[i].resident_sample)
                            continue;
                        if (victim == SIZE_MAX || windows[i].recency < windows[victim].recency)
                            victim = i;
                    }
                    if (victim == SIZE_MAX)
                        break;
                    error = release_window(&windows[victim]);
                    if (error)
                        break;
                    size_t after = read_rss();
                    if (after == SIZE_MAX) {
                        error = EIO;
                        break;
                    }
                    if (rss > after)
                        total_reclaimed += rss - after;
                    rss = after;
                }
            }
            pthread_mutex_unlock(&map_lock);
            governor_cpu_ns += time_ns(CLOCK_THREAD_CPUTIME_ID) - cpu_start;
            affinity(0);
            rss = read_rss();
            if (requested == purge_epoch && !error)
                acknowledge(purge_epoch, rss);
        }
        double now = time_ns(CLOCK_MONOTONIC) * 1e-9;
        if (error || total_evictions != before_count || now - last_log >= 1) {
            pthread_mutex_lock(&map_lock);
            size_t observed_windows = count;
            pthread_mutex_unlock(&map_lock);
            fprintf(log,
                    "{\"time_monotonic\":%.6f,\"pid\":%d,\"rss_bytes\":%zu,\"peak_rss_bytes\":%llu,"
                    "\"budget_bytes\":%zu,\"headroom_bytes\":%zu,\"windows\":%zu,\"evictions\":%"
                    "llu,\"reclaimed_rss_bytes\":%llu,\"governor_cpu_ns\":%llu,\"purge_epoch\":%"
                    "llu,\"error\":%d}\n",
                    now, getpid(), rss, (unsigned long long)peak_rss, budget, margin,
                    observed_windows, (unsigned long long)total_evictions,
                    (unsigned long long)total_reclaimed, (unsigned long long)governor_cpu_ns,
                    (unsigned long long)purge_epoch, error);
            last_log = now;
        }
        if (error)
            _exit(96);
    }
    return NULL;
}

__attribute__((constructor)) static void initialize(void) {
    const char *model = getenv("TIDEWAY_LLAMA_MODEL");
    const char *b = getenv("TIDEWAY_LLAMA_RSS_BUDGET_BYTES");
    if (!model || !b || !(budget = strtoull(b, NULL, 10)))
        return;
    const char *parent = getenv("TIDEWAY_LLAMA_EXPECTED_PARENT");
    if (parent &&
        (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != (pid_t)strtol(parent, NULL, 10)))
        _exit(100);
    const char *l = getenv("TIDEWAY_LLAMA_BUDGET_LOG");
    const char *c = getenv("TIDEWAY_LLAMA_BUDGET_CONTROL");
    const char *a = getenv("TIDEWAY_LLAMA_BUDGET_ACK");
    if (!l || !c || !a)
        _exit(97);
    snprintf(log_path, sizeof(log_path), "%s", l);
    snprintf(control_path, sizeof(control_path), "%s", c);
    snprintf(ack_path, sizeof(ack_path), "%s", a);
    page_bytes = sysconf(_SC_PAGE_SIZE);
    margin = getenv("TIDEWAY_LLAMA_HEADROOM_BYTES")
                 ? strtoull(getenv("TIDEWAY_LLAMA_HEADROOM_BYTES"), NULL, 10)
                 : 1ull << 30;
    window_bytes = getenv("TIDEWAY_LLAMA_WINDOW_BYTES")
                       ? strtoull(getenv("TIDEWAY_LLAMA_WINDOW_BYTES"), NULL, 10)
                       : 512ull << 20;
    if (!window_bytes || window_bytes % page_bytes || margin * 2 >= budget)
        _exit(97);
    struct stat s;
    target_fd = open(model, O_RDONLY | O_CLOEXEC);
    if (target_fd < 0 || fstat(target_fd, &s) || !S_ISREG(s.st_mode))
        _exit(97);
    target_dev = s.st_dev;
    target_ino = s.st_ino;
    real_posix_madvise = dlsym(RTLD_NEXT, "posix_madvise");
    active = 1;
    pthread_t thread;
    if (pthread_create(&thread, NULL, govern, NULL))
        _exit(98);
    pthread_detach(thread);
}
