/* CUDA driver VMM storage with DLPack ownership, used only by Tideway.
 * No CUDA kernel or runtime allocator is involved. Each committed physical
 * chunk has exactly one VA owner. Python fences GPU work before changing maps.
 * Tensor and controller references independently keep reservations alive.
 */
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

typedef struct {
    int device_type, device_id;
} DLDevice;
typedef struct {
    uint8_t code, bits;
    uint16_t lanes;
} DLDataType;
typedef struct {
    int device;
    size_t chunk, budget, capacity, owned, mapped, free_count;
    CUmemGenericAllocationHandle *free_handles;
    uint64_t creates, releases, maps, unmaps, accesses, reuses;
    _Atomic int refs;
    pthread_mutex_t mutex;
} PhysicalPool;

typedef struct {
    void *data;
    DLDevice device;
    int ndim;
    DLDataType dtype;
    int64_t *shape, *strides;
    uint64_t byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *);
} DLManagedTensor;

static void *driver;
static char error_msg[512];
#define DRIVER_FN(name) static __typeof__(&name) p_##name
DRIVER_FN(cuMemAddressReserve);
DRIVER_FN(cuMemAddressFree);
DRIVER_FN(cuMemCreate);
DRIVER_FN(cuMemRelease);
DRIVER_FN(cuMemMap);
DRIVER_FN(cuMemUnmap);
DRIVER_FN(cuMemSetAccess);
DRIVER_FN(cuMemGetAllocationGranularity);
DRIVER_FN(cuGetErrorName);
DRIVER_FN(cuDeviceGetAttribute);

typedef struct {
    CUdeviceptr address;
    size_t size, chunk, n_chunks;
    int device;
    _Atomic int refs;
    CUmemGenericAllocationHandle *handles;
    PhysicalPool *pool;
} Region;

static int fail(const char *where, CUresult result) {
    const char *name = "unknown CUDA error";
    if (p_cuGetErrorName)
        p_cuGetErrorName(result, &name);
    snprintf(error_msg, sizeof(error_msg), "%s: %s (%d)", where, name, (int)result);
    return -1;
}
const char *tideway_vmm_error(void) { return error_msg; }

int tideway_vmm_init(int device) {
    if (!driver) {
        driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!driver) {
            snprintf(error_msg, sizeof(error_msg), "%s", dlerror());
            return -1;
        }
#define LOAD(name)                                                                                 \
    do {                                                                                           \
        *(void **)(&p_##name) = dlsym(driver, #name);                                              \
        if (!p_##name) {                                                                           \
            snprintf(error_msg, sizeof(error_msg), "missing driver API %s", #name);                \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)
        LOAD(cuMemAddressReserve);
        LOAD(cuMemAddressFree);
        LOAD(cuMemCreate);
        LOAD(cuMemRelease);
        LOAD(cuMemMap);
        LOAD(cuMemUnmap);
        LOAD(cuMemSetAccess);
        LOAD(cuMemGetAllocationGranularity);
        LOAD(cuGetErrorName);
        LOAD(cuDeviceGetAttribute);
#undef LOAD
    }
    int supported = 0;
    CUresult rc = p_cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);
    if (rc != CUDA_SUCCESS)
        return fail("cuDeviceGetAttribute", rc);
    if (!supported) {
        snprintf(error_msg, sizeof(error_msg), "device has no CUDA VMM support");
        return -1;
    }
    return 0;
}

void *tideway_vmm_pool_create(int device, size_t chunk, size_t budget) {
    if (!chunk || budget < chunk) {
        snprintf(error_msg, sizeof(error_msg), "invalid VMM physical pool budget");
        return NULL;
    }
    PhysicalPool *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->capacity = budget / chunk;
    p->free_handles = calloc(p->capacity, sizeof(*p->free_handles));
    if (!p->free_handles) {
        free(p);
        return NULL;
    }
    p->device = device;
    p->chunk = chunk;
    p->budget = budget;
    p->refs = 1;
    pthread_mutex_init(&p->mutex, NULL);
    return p;
}

void tideway_vmm_pool_release(PhysicalPool *p) {
    if (!p || atomic_fetch_sub_explicit(&p->refs, 1, memory_order_acq_rel) > 1)
        return;
    /* Every Region has a reference. Thus the final reference can disappear
       only when all physical handles are idle and no mapping is still live. */
    if (p->mapped || p->free_count != p->owned) {
        snprintf(error_msg, sizeof(error_msg), "VMM pool destroyed with live mappings");
        return; /* Quarantine ownership; never free physical storage still used. */
    }
    while (p->free_count) {
        CUresult rc = p_cuMemRelease(p->free_handles[p->free_count - 1]);
        if (rc != CUDA_SUCCESS) {
            fail("cuMemRelease pooled teardown", rc);
            return;
        }
        p->free_count--;
        p->owned--;
        p->releases++;
    }
    pthread_mutex_destroy(&p->mutex);
    free(p->free_handles);
    free(p);
}

void tideway_vmm_pool_stats(PhysicalPool *p, uint64_t *out) {
    pthread_mutex_lock(&p->mutex);
    out[0] = p->budget;
    out[1] = p->owned * p->chunk;
    out[2] = p->mapped * p->chunk;
    out[3] = p->free_count * p->chunk;
    out[4] = p->creates;
    out[5] = p->releases;
    out[6] = p->maps;
    out[7] = p->unmaps;
    out[8] = p->accesses;
    out[9] = p->reuses;
    pthread_mutex_unlock(&p->mutex);
}

void *tideway_vmm_reserve(size_t size, int device, size_t requested_chunk) {
    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t minimum;
    CUresult rc =
        p_cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (rc != CUDA_SUCCESS) {
        fail("cuMemGetAllocationGranularity", rc);
        return NULL;
    }
    size_t chunk = ((requested_chunk + minimum - 1) / minimum) * minimum;
    size_t aligned = ((size + chunk - 1) / chunk) * chunk;
    Region *r = calloc(1, sizeof(*r));
    if (!r)
        return NULL;
    r->size = aligned;
    r->chunk = chunk;
    r->n_chunks = aligned / chunk;
    r->device = device;
    r->refs = 1;
    r->handles = calloc(r->n_chunks, sizeof(*r->handles));
    if (!r->handles) {
        free(r);
        return NULL;
    }
    rc = p_cuMemAddressReserve(&r->address, aligned, chunk, 0, 0);
    if (rc != CUDA_SUCCESS) {
        fail("cuMemAddressReserve", rc);
        free(r->handles);
        free(r);
        return NULL;
    }
    return r;
}
uint64_t tideway_vmm_address(Region *r) { return r->address; }
size_t tideway_vmm_chunk(Region *r) { return r->chunk; }

void *tideway_vmm_reserve_pooled(size_t size, PhysicalPool *p) {
    Region *r = tideway_vmm_reserve(size, p->device, p->chunk);
    if (!r)
        return NULL;
    if (r->chunk != p->chunk) {
        p_cuMemAddressFree(r->address, r->size);
        free(r->handles);
        free(r);
        snprintf(error_msg, sizeof(error_msg), "pool chunk differs from driver granularity");
        return NULL;
    }
    r->pool = p;
    atomic_fetch_add_explicit(&p->refs, 1, memory_order_relaxed);
    return r;
}

static int pooled_set(Region *r, size_t index, int commit) {
    PhysicalPool *p = r->pool;
    CUdeviceptr at = r->address + index * r->chunk;
    CUresult rc;
    int result = 0;
    pthread_mutex_lock(&p->mutex);
    if (!commit) {
        rc = p_cuMemUnmap(at, r->chunk);
        if (rc != CUDA_SUCCESS) {
            result = fail("cuMemUnmap pooled", rc);
            goto done;
        }
        p->unmaps++;
        p->mapped--;
        p->free_handles[p->free_count++] = r->handles[index];
        r->handles[index] = 0;
        goto done;
    }
    CUmemGenericAllocationHandle h;
    if (p->free_count) {
        h = p->free_handles[--p->free_count];
        p->reuses++;
    } else {
        if (p->owned >= p->capacity) {
            snprintf(error_msg, sizeof(error_msg), "VMM physical ownership budget exhausted");
            result = -1;
            goto done;
        }
        CUmemAllocationProp prop = {0};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = p->device;
        rc = p_cuMemCreate(&h, p->chunk, &prop, 0);
        if (rc != CUDA_SUCCESS) {
            result = fail("cuMemCreate pooled", rc);
            goto done;
        }
        p->creates++;
        p->owned++;
    }
    rc = p_cuMemMap(at, p->chunk, 0, h, 0);
    if (rc != CUDA_SUCCESS) {
        p->free_handles[p->free_count++] = h;
        result = fail("cuMemMap pooled", rc);
        goto done;
    }
    /* Once mapped, record the true VA owner before any further API can fail.
       A failed access/rollback never loses the only physical-handle reference. */
    r->handles[index] = h;
    p->mapped++;
    p->maps++;
    CUmemAccessDesc access = {0};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = p->device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = p_cuMemSetAccess(at, p->chunk, &access, 1);
    if (rc != CUDA_SUCCESS) {
        result = fail("cuMemSetAccess pooled", rc);
        CUresult undo = p_cuMemUnmap(at, p->chunk);
        if (undo == CUDA_SUCCESS) {
            p->unmaps++;
            p->mapped--;
            r->handles[index] = 0;
            p->free_handles[p->free_count++] = h;
        }
        /* A failed rollback leaves r owning h and its mapping. The controller
           latches failed; teardown can retry, and owned bytes stay budgeted. */
        goto done;
    }
    p->accesses++;
done:
    pthread_mutex_unlock(&p->mutex);
    return result;
}

int tideway_vmm_set(Region *r, size_t index, int commit) {
    if (index >= r->n_chunks) {
        snprintf(error_msg, sizeof(error_msg), "VMM chunk outside reservation");
        return -1;
    }
    if ((r->handles[index] != 0) == (commit != 0))
        return 0;
    if (r->pool)
        return pooled_set(r, index, commit);
    CUdeviceptr at = r->address + index * r->chunk;
    CUresult rc;
    if (!commit) {
        rc = p_cuMemUnmap(at, r->chunk);
        if (rc != CUDA_SUCCESS)
            return fail("cuMemUnmap", rc);
        rc = p_cuMemRelease(r->handles[index]);
        if (rc != CUDA_SUCCESS)
            return fail("cuMemRelease", rc);
        r->handles[index] = 0;
        return 0;
    }
    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = r->device;
    CUmemGenericAllocationHandle h;
    rc = p_cuMemCreate(&h, r->chunk, &prop, 0);
    if (rc != CUDA_SUCCESS)
        return fail("cuMemCreate", rc);
    rc = p_cuMemMap(at, r->chunk, 0, h, 0);
    if (rc != CUDA_SUCCESS) {
        p_cuMemRelease(h);
        return fail("cuMemMap", rc);
    }
    CUmemAccessDesc access = {0};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = p_cuMemSetAccess(at, r->chunk, &access, 1);
    if (rc != CUDA_SUCCESS) {
        p_cuMemUnmap(at, r->chunk);
        p_cuMemRelease(h);
        return fail("cuMemSetAccess", rc);
    }
    r->handles[index] = h;
    return 0;
}
void tideway_vmm_release(Region *r) {
    if (!r || atomic_fetch_sub_explicit(&r->refs, 1, memory_order_acq_rel) > 1)
        return;
    for (size_t i = 0; i < r->n_chunks; i++) {
        if (r->handles[i]) {
            if (r->pool) {
                /* If the driver cannot drain the mapping, quarantine this
                   Region (and its pool ref) rather than invalidate live VA. */
                if (pooled_set(r, i, 0) < 0)
                    return;
            } else {
                p_cuMemUnmap(r->address + i * r->chunk, r->chunk);
                p_cuMemRelease(r->handles[i]);
            }
        }
    }
    p_cuMemAddressFree(r->address, r->size);
    PhysicalPool *pool = r->pool;
    free(r->handles);
    free(r);
    tideway_vmm_pool_release(pool);
}
static void tensor_delete(DLManagedTensor *tensor) {
    tideway_vmm_release(tensor->manager_ctx);
    free(tensor->dl_tensor.shape);
    free(tensor);
}
void tideway_vmm_dlpack_release(DLManagedTensor *tensor) { tensor_delete(tensor); }
void *tideway_vmm_dlpack(Region *r, int ndim, const int64_t *shape, int code, int bits) {
    DLManagedTensor *t = calloc(1, sizeof(*t));
    if (!t)
        return NULL;
    t->dl_tensor.shape = malloc(sizeof(int64_t) * (size_t)ndim);
    if (!t->dl_tensor.shape) {
        free(t);
        return NULL;
    }
    memcpy(t->dl_tensor.shape, shape, sizeof(int64_t) * (size_t)ndim);
    t->dl_tensor.data = (void *)(uintptr_t)r->address;
    t->dl_tensor.device = (DLDevice){2, r->device}; /* kDLCUDA */
    t->dl_tensor.ndim = ndim;
    t->dl_tensor.dtype = (DLDataType){(uint8_t)code, (uint8_t)bits, 1};
    t->manager_ctx = r;
    t->deleter = tensor_delete;
    atomic_fetch_add_explicit(&r->refs, 1, memory_order_relaxed);
    return t;
}
