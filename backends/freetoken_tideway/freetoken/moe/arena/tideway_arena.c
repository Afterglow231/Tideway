/* Tideway-Shift elastic region (v2): a small first-fit allocator over fixed device address
 * ranges -- the tail rows of the expert bank caches -- used through
 * torch.cuda.memory.CUDAPluggableAllocator + torch.cuda.MemPool so that a prefill batch's
 * temporaries live in rows that are ordinary LRU expert slots during decode.
 *
 *  - per-region HEAD (the first row): reserved for small (<= 2 MiB) segments, so the few
 *    long-lived tiny tensors PyTorch keeps around never share a row with an expert slot
 *    (the head row is excluded from the LRU on the Python side);
 *  - per-region LIMIT: a prefill is confined to [head, limit) so a small prefill touches --
 *    and the Python side invalidates -- only the first few rows;
 *  - cudaMalloc FALLBACK when the limited area is exhausted: the forward never fails; the
 *    fallback bytes are counted so the Python size model raises its estimate;
 *  - high-water tracking per region (bytes from base) since the last reset.
 * Single-threaded use (the scheduler process). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#define MAXR 4
#define MAXB 8192
#define MAXF 1024
typedef struct {
    uintptr_t addr;
    size_t size;
    int free;
} Blk;
typedef struct {
    uintptr_t base;
    size_t cap, head, limit, hw;
    Blk b[MAXB];
    int n;
} Reg;
static Reg R[MAXR];
static int nreg = 0;
static size_t in_use = 0, peak = 0;
static long n_alloc = 0, n_free = 0, n_fail = 0, n_small_spill = 0;
static struct {
    void *p;
    size_t sz;
} F[MAXF];
static int nf = 0;
static size_t fb_in_use = 0, fb_peak = 0;
static long n_fb = 0, n_fb_fail = 0;
static const size_t ALIGN = (size_t)2 << 20;
typedef int (*cudaMalloc_t)(void **, size_t);
typedef int (*cudaFree_t)(void *);
static cudaMalloc_t p_cudaMalloc = NULL;
static cudaFree_t p_cudaFree = NULL;

void tideway_arena_set(int idx, uintptr_t base, size_t cap, size_t head) {
    if (idx < 0 || idx >= MAXR)
        return;
    Reg *r = &R[idx];
    r->base = base;
    r->cap = cap;
    r->head = head > cap ? cap : head;
    r->limit = cap;
    r->hw = 0;
    r->n = 1;
    r->b[0].addr = base;
    r->b[0].size = cap;
    r->b[0].free = 1;
    if (idx + 1 > nreg)
        nreg = idx + 1;
}
void tideway_arena_limit(int idx, size_t limit) {
    if (idx < 0 || idx >= nreg)
        return;
    Reg *r = &R[idx];
    if (limit < r->head)
        limit = r->head;
    if (limit > r->cap)
        limit = r->cap;
    r->limit = limit;
}
void tideway_arena_reset_hw(void) {
    for (int i = 0; i < nreg; i++)
        R[i].hw = 0;
    fb_peak = fb_in_use;
}
/* out[0] in_use  [1] peak  [2] n_alloc  [3] n_free  [4] n_fail (arena misses -> fallback tried)
 * [5] fb_in_use  [6] fb_peak (since reset_hw)  [7] n_fb  [8..11] hw per region  [12] n_small_spill
 * [13] n_fb_fail */
void tideway_arena_stats(size_t *out) {
    out[0] = in_use;
    out[1] = peak;
    out[2] = (size_t)n_alloc;
    out[3] = (size_t)n_free;
    out[4] = (size_t)n_fail;
    out[5] = fb_in_use;
    out[6] = fb_peak;
    out[7] = (size_t)n_fb;
    for (int i = 0; i < 4; i++)
        out[8 + i] = i < nreg ? R[i].hw : 0;
    out[12] = (size_t)n_small_spill;
    out[13] = (size_t)n_fb_fail;
}

/* split free block i at address `at` (strictly inside it); returns the index of the second part */
static int split_at(Reg *r, int i, uintptr_t at) {
    Blk *b = &r->b[i];
    if (r->n >= MAXB)
        return -1;
    memmove(&r->b[i + 2], &r->b[i + 1], sizeof(Blk) * (size_t)(r->n - i - 1));
    r->b[i + 1].addr = at;
    r->b[i + 1].size = b->addr + b->size - at;
    r->b[i + 1].free = 1;
    b->size = at - b->addr;
    r->n++;
    return i + 1;
}
static void *take(Reg *r, int i, size_t size, int main_area) {
    Blk *b = &r->b[i];
    if (b->size > size) {
        if (split_at(r, i, b->addr + size) < 0)
            return NULL;
        b = &r->b[i];
    }
    b->free = 0;
    in_use += size;
    if (in_use > peak)
        peak = in_use;
    n_alloc++;
    if (main_area) {
        size_t off = (size_t)(b->addr + size - r->base);
        if (off > r->hw)
            r->hw = off;
    }
    return (void *)b->addr;
}
/* first free block that can hold [a, a + size) inside [lo, hi), a = max(block start, lo) */
static void *fit(Reg *r, size_t size, uintptr_t lo, uintptr_t hi, int main_area) {
    for (int i = 0; i < r->n; i++) {
        Blk *b = &r->b[i];
        if (!b->free)
            continue;
        uintptr_t a = b->addr < lo ? lo : b->addr;
        uintptr_t end = b->addr + b->size;
        if (a + size > hi || a + size > end)
            continue;
        if (a > b->addr) {
            int j = split_at(r, i, a);
            if (j < 0)
                return NULL;
            i = j;
        }
        return take(r, i, size, main_area);
    }
    return NULL;
}
void *tideway_alloc(size_t size, int device, void *stream) {
    (void)device;
    (void)stream;
    size = (size + ALIGN - 1) & ~(ALIGN - 1);
    void *p;
    if (size <= ALIGN)
        for (int i = 0; i < nreg; i++)
            if ((p = fit(&R[i], size, R[i].base, R[i].base + R[i].head, 0)))
                return p;
    for (int i = 0; i < nreg; i++)
        if ((p = fit(&R[i], size, R[i].base + R[i].head, R[i].base + R[i].limit, 1))) {
            if (size <= ALIGN)
                n_small_spill++;
            return p;
        }
    n_fail++;
    if (!p_cudaMalloc || !p_cudaFree) {
        p_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_DEFAULT, "cudaMalloc");
        p_cudaFree = (cudaFree_t)dlsym(RTLD_DEFAULT, "cudaFree");
        if (!p_cudaMalloc || !p_cudaFree) {
            void *h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_GLOBAL);
            if (!h)
                h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
            if (h) {
                p_cudaMalloc = (cudaMalloc_t)dlsym(h, "cudaMalloc");
                p_cudaFree = (cudaFree_t)dlsym(h, "cudaFree");
            }
        }
    }
    if (p_cudaMalloc && p_cudaFree && nf < MAXF) {
        void *q = NULL;
        if (p_cudaMalloc(&q, size) == 0 && q) {
            F[nf].p = q;
            F[nf].sz = size;
            nf++;
            fb_in_use += size;
            if (fb_in_use > fb_peak)
                fb_peak = fb_in_use;
            n_fb++;
            return q;
        }
    }
    n_fb_fail++;
    return NULL;
}
void tideway_free(void *ptr, size_t size, int device, void *stream) {
    (void)size;
    (void)device;
    (void)stream;
    uintptr_t a = (uintptr_t)ptr;
    for (int k = 0; k < nreg; k++) {
        Reg *r = &R[k];
        if (a < r->base || a >= r->base + r->cap)
            continue;
        for (int i = 0; i < r->n; i++) {
            if (r->b[i].addr == a && !r->b[i].free) {
                r->b[i].free = 1;
                in_use -= r->b[i].size;
                n_free++;
                if (i + 1 < r->n && r->b[i + 1].free) {
                    r->b[i].size += r->b[i + 1].size;
                    memmove(&r->b[i + 1], &r->b[i + 2], sizeof(Blk) * (size_t)(r->n - i - 2));
                    r->n--;
                }
                if (i > 0 && r->b[i - 1].free) {
                    r->b[i - 1].size += r->b[i].size;
                    memmove(&r->b[i], &r->b[i + 1], sizeof(Blk) * (size_t)(r->n - i - 1));
                    r->n--;
                }
                return;
            }
        }
        return;
    }
    for (int i = 0; i < nf; i++)
        if (F[i].p == ptr) {
            fb_in_use -= F[i].sz;
            F[i] = F[nf - 1];
            nf--;
            if (p_cudaFree)
                p_cudaFree(ptr);
            return;
        }
}
