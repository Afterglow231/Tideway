#include <cuda_runtime.h>
#include <stdint.h>

struct OutputMap {
    uint8_t *base[4];
    uint64_t offset[4];
    int count;
};
__device__ void store_byte(OutputMap map, uint64_t pos, uint8_t value) {
    int bank = map.count - 1;
    while (bank > 0 && pos < map.offset[bank])
        --bank;
    map.base[bank][pos - map.offset[bank]] = value;
}

/* Descriptor: source offset/length, output offset/count/stride, table, raw flag, checksum. */
__global__ void decode_rows(const uint8_t *input, const uint32_t *descs, int n,
                            const uint32_t *freqs, const uint32_t *cums, const uint8_t *lookups,
                            OutputMap output, int *errors) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;
    const uint32_t *d = descs + (int64_t)row * 8;
    const uint8_t *src = input + d[0];
    uint32_t count = (uint32_t)d[3], stride = (uint32_t)d[4];
    uint32_t a = 1, b = 0;
    if (d[6] & 1) {
        for (uint32_t i = 0; i < count; ++i) {
            uint8_t value = src[i];
            store_byte(output, d[2] + (uint64_t)i * stride, value);
            a += value;
            b += a;
        }
    } else {
        const uint32_t *freq = freqs + d[5] * 256, *cum = cums + d[5] * 256;
        const uint8_t *lut = lookups + d[5] * 4096;
        if (d[1] < 4) {
            atomicAdd(errors, 1);
            return;
        }
        uint32_t state = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
                         ((uint32_t)src[3] << 24);
        uint64_t at = 4;
        bool nibble = (d[6] & 2) != 0;
        uint32_t low = 0;
        for (uint32_t i = 0; i < count * (nibble ? 2 : 1); ++i) {
            uint32_t slot = state & 4095, symbol = lut[slot];
            if (!nibble) {
                store_byte(output, d[2] + (uint64_t)i * stride, (uint8_t)symbol);
                a += symbol;
                b += a;
            } else if (!(i & 1))
                low = symbol;
            else {
                uint32_t value = low | (symbol << 4);
                store_byte(output, d[2] + (uint64_t)(i / 2) * stride, (uint8_t)value);
                a += value;
                b += a;
            }
            state = freq[symbol] * (state >> 12) + slot - cum[symbol];
            while (state < (1u << 23)) {
                if (at >= (uint64_t)d[1]) {
                    atomicAdd(errors, 1);
                    return;
                }
                state = (state << 8) | src[at++];
            }
        }
        if (at != (uint64_t)d[1] || state != (1u << 23))
            atomicAdd(errors, 1);
    }
    uint32_t check = a ^ ((b << 16) | (b >> 16));
    if (check != (uint32_t)d[7])
        atomicAdd(errors, 1);
}

extern "C" int rans_cuda_decode(const void *input, const void *descs, int n, const void *freqs,
                                const void *cums, const void *lookups, const uint64_t *pointers,
                                const uint64_t *offsets, int n_banks, void *errors, void *stream) {
    if (n_banks < 1 || n_banks > 4)
        return (int)cudaErrorInvalidValue;
    OutputMap output;
    output.count = n_banks;
    for (int i = 0; i < n_banks; ++i) {
        output.base[i] = (uint8_t *)pointers[i];
        output.offset[i] = offsets[i];
    }
    decode_rows<<<(n + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t *)input, (const uint32_t *)descs, n, (const uint32_t *)freqs,
        (const uint32_t *)cums, (const uint8_t *)lookups, output, (int *)errors);
    return (int)cudaGetLastError();
}
