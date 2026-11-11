/* Independent byte-rANS codec. Fixed 12-bit frequency normalization; no lossy transform. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define SCALE 12
#define TOTAL (1u << SCALE)
#define LOWER (1u << 23)

void rans_histogram(const uint8_t *src, uint64_t count, uint32_t stride, uint64_t *hist) {
    memset(hist, 0, 256 * sizeof(uint64_t));
    for (uint64_t i = 0; i < count; ++i)
        ++hist[src[i * stride]];
}
void rans_histogram_nibbles(const uint8_t *src, uint64_t count, uint32_t stride, uint64_t *hist) {
    memset(hist, 0, 256 * sizeof(uint64_t));
    for (uint64_t i = 0; i < count; ++i) {
        uint8_t v = src[i * stride];
        ++hist[v & 15];
        ++hist[v >> 4];
    }
}

/* Encode one independently decodable stream backwards into caller-owned memory. */
static int64_t encode(const uint8_t *src, uint32_t count, uint32_t stride, const uint32_t *freq,
                      const uint32_t *cum, uint8_t *dst, uint64_t capacity, int nibble) {
    uint64_t at = capacity;
    uint32_t state = LOWER;
    for (uint32_t i = count * (nibble ? 2 : 1); i; --i) {
        uint32_t at_symbol = i - 1;
        uint32_t symbol =
            nibble ? ((src[(uint64_t)(at_symbol / 2) * stride] >> ((at_symbol & 1) * 4)) & 15)
                   : src[(uint64_t)at_symbol * stride];
        uint32_t f = freq[symbol];
        if (!f)
            return -1;
        uint32_t limit = ((LOWER >> SCALE) << 8) * f;
        while (state >= limit) {
            if (at <= 4)
                return -2;
            dst[--at] = (uint8_t)state;
            state >>= 8;
        }
        state = ((state / f) << SCALE) + (state % f) + cum[symbol];
    }
    if (at < 4)
        return -3;
    at -= 4;
    for (unsigned i = 0; i < 4; ++i)
        dst[at + i] = (uint8_t)(state >> (8 * i));
    uint64_t size = capacity - at;
    memmove(dst, dst + at, size);
    return (int64_t)size;
}
int64_t rans_encode(const uint8_t *s, uint32_t n, uint32_t st, const uint32_t *f, const uint32_t *c,
                    uint8_t *d, uint64_t cap) {
    return encode(s, n, st, f, c, d, cap, 0);
}
int64_t rans_encode_nibbles(const uint8_t *s, uint32_t n, uint32_t st, const uint32_t *f,
                            const uint32_t *c, uint8_t *d, uint64_t cap) {
    return encode(s, n, st, f, c, d, cap, 1);
}

static int decode(const uint8_t *src, uint64_t size, uint8_t *dst, uint32_t count, uint32_t stride,
                  const uint32_t *freq, const uint32_t *cum, const uint8_t *lookup, int nibble) {
    if (size < 4)
        return 1;
    uint32_t state = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
                     ((uint32_t)src[3] << 24);
    uint64_t at = 4;
    uint32_t low = 0;
    for (uint32_t i = 0; i < count * (nibble ? 2 : 1); ++i) {
        uint32_t slot = state & (TOTAL - 1), symbol = lookup[slot];
        if (!nibble)
            dst[(uint64_t)i * stride] = (uint8_t)symbol;
        else if (!(i & 1))
            low = symbol;
        else
            dst[(uint64_t)(i / 2) * stride] = (uint8_t)(low | (symbol << 4));
        state = freq[symbol] * (state >> SCALE) + slot - cum[symbol];
        while (state < LOWER) {
            if (at >= size)
                return 2;
            state = (state << 8) | src[at++];
        }
    }
    return (at == size && state == LOWER) ? 0 : 3;
}
int rans_decode(const uint8_t *s, uint64_t size, uint8_t *d, uint32_t n, uint32_t st,
                const uint32_t *f, const uint32_t *c, const uint8_t *l) {
    return decode(s, size, d, n, st, f, c, l, 0);
}
int rans_decode_nibbles(const uint8_t *s, uint64_t size, uint8_t *d, uint32_t n, uint32_t st,
                        const uint32_t *f, const uint32_t *c, const uint8_t *l) {
    return decode(s, size, d, n, st, f, c, l, 1);
}

uint32_t bytes_checksum(const uint8_t *src, uint32_t count, uint32_t stride) {
    uint32_t a = 1, b = 0;
    for (uint32_t i = 0; i < count; ++i) {
        a += src[(uint64_t)i * stride];
        b += a;
    }
    return a ^ ((b << 16) | (b >> 16));
}
