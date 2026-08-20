<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# hip_helper.h — Drop-in Replacement for helper_cuda.h

This file contains the source for `hip_helper.h`, the AMD HIP equivalent of
NVIDIA's `helper_cuda.h`. **Create this file before running hipify** — builds
will fail without it (S40).

## When to create it

After hipify-perl converts `#include "helper_cuda.h"` to `#include "hip_helper.h"`,
the file must exist in the project's `Common/` directory. Create it with:

```bash
# From the skill reference directory
# Ask the skill to create it from reference/hip_helper.md
# or write the code block below manually to Common/hip_helper.h

# Or write it directly from the content below
```

## Source

Write the following content to `Common/hip_helper.h` in your project:

```cpp
/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_HELPER_H
#define HIP_HELPER_H

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Error Checking
// ============================================================================

#ifndef HIP_CHECK
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

#ifndef checkCudaErrors
#define checkCudaErrors(call) HIP_CHECK(call)
#endif

#ifndef checkHipErrors
#define checkHipErrors(call) HIP_CHECK(call)
#endif

#ifndef getLastCudaError
inline void getLastCudaError(const char *msg) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error: %s: %s\n", msg, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#endif

// ============================================================================
// Library-specific Error Checking
// ============================================================================

// hipFFT error checking
#ifdef __HIPCC__
#include <hipfft/hipfft.h>
#ifndef HIPFFT_CHECK
#define HIPFFT_CHECK(call) \
    do { \
        hipfftResult result = call; \
        if (result != HIPFFT_SUCCESS) { \
            fprintf(stderr, "hipFFT error at %s:%d: code=%d\n", \
                    __FILE__, __LINE__, result); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif
#endif

// ============================================================================
// Device Selection (only define if not already defined by helper_cuda.h)
// ============================================================================

#ifndef __HELPER_CUDA_H__

inline int findCudaDevice(int argc, const char **argv) {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No HIP devices found!\n");
        exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipSetDevice(0));
    return 0;
}

inline int gpuDeviceInit(int devID) {
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No HIP devices found!\n");
        exit(EXIT_FAILURE);
    }
    if (devID < 0) devID = 0;
    if (devID >= device_count) devID = device_count - 1;
    HIP_CHECK(hipSetDevice(devID));
    return devID;
}

inline int gpuGetMaxGflopsDeviceId() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) return -1;
    int max_gflops_device = 0;
    unsigned long long max_gflops = 0;
    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        unsigned long long gflops = (unsigned long long)props.multiProcessorCount * props.clockRate;
        if (gflops > max_gflops) {
            max_gflops = gflops;
            max_gflops_device = i;
        }
    }
    return max_gflops_device;
}

inline bool checkCudaCapabilities(int major, int minor) {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    return deviceCount > 0;
}

inline int _ConvertSMVer2Cores(int major, int minor) {
    return 64;  // typical wavefront size
}

#endif // __HELPER_CUDA_H__

// ============================================================================
// System Atomics Fallbacks
// ============================================================================

#ifndef atomicAdd_system
#define atomicAdd_system atomicAdd
#endif

#ifndef atomicInc_system
__device__ __forceinline__ unsigned int atomicInc_system(unsigned int *addr, unsigned int val) {
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, (assumed >= val) ? 0 : assumed + 1);
    } while (assumed != old);
    return old;
}
#endif

#ifndef atomicDec_system
__device__ __forceinline__ unsigned int atomicDec_system(unsigned int *addr, unsigned int val) {
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, ((assumed == 0) || (assumed > val)) ? val : assumed - 1);
    } while (assumed != old);
    return old;
}
#endif

// ============================================================================
// Shared Memory Helper
// ============================================================================

#ifndef SHAREDMEM_DEFINED
template <typename T>
class SharedMemory {
public:
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};
#endif // SHAREDMEM_DEFINED

// ============================================================================
// Command Line Parsing (helper_string.h equivalents)
// ============================================================================

#ifndef __HELPER_STRING_H__

#include <string.h>

inline bool checkCmdLineFlag(int argc, const char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) {
        if (strstr(argv[i], flag) != NULL) {
            return true;
        }
    }
    return false;
}

inline int getCmdLineArgumentInt(int argc, const char **argv, const char *flag) {
    char search[256];
    snprintf(search, sizeof(search), "%s=", flag);
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], search, strlen(search)) == 0) {
            return atoi(argv[i] + strlen(search));
        }
    }
    return 0;
}

inline bool getCmdLineArgumentString(int argc, const char **argv,
                                     const char *flag, char **value) {
    char search[256];
    snprintf(search, sizeof(search), "%s=", flag);
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], search, strlen(search)) == 0) {
            *value = (char *)(argv[i] + strlen(search));
            return true;
        }
    }
    return false;
}

#endif // __HELPER_STRING_H__

// ============================================================================
// Timer Interface (helper_timer.h equivalents)
// ============================================================================

#ifndef __HELPER_TIMER_H__

#include <sys/time.h>

typedef struct StopWatchInterface {
    struct timeval start_time;
    struct timeval end_time;
    float total_time;
    bool running;
} StopWatchInterface;

inline void sdkCreateTimer(StopWatchInterface **timer) {
    *timer = (StopWatchInterface *)malloc(sizeof(StopWatchInterface));
    (*timer)->total_time = 0.0f;
    (*timer)->running = false;
}

inline void sdkDeleteTimer(StopWatchInterface **timer) {
    if (*timer) {
        free(*timer);
        *timer = NULL;
    }
}

inline void sdkStartTimer(StopWatchInterface **timer) {
    gettimeofday(&(*timer)->start_time, NULL);
    (*timer)->running = true;
}

inline void sdkStopTimer(StopWatchInterface **timer) {
    gettimeofday(&(*timer)->end_time, NULL);
    (*timer)->running = false;
    float elapsed = ((*timer)->end_time.tv_sec - (*timer)->start_time.tv_sec) * 1000.0f +
                    ((*timer)->end_time.tv_usec - (*timer)->start_time.tv_usec) / 1000.0f;
    (*timer)->total_time += elapsed;
}

inline void sdkResetTimer(StopWatchInterface **timer) {
    (*timer)->total_time = 0.0f;
    (*timer)->running = false;
}

inline float sdkGetTimerValue(StopWatchInterface **timer) {
    if ((*timer)->running) {
        struct timeval now;
        gettimeofday(&now, NULL);
        float elapsed = (now.tv_sec - (*timer)->start_time.tv_sec) * 1000.0f +
                        (now.tv_usec - (*timer)->start_time.tv_usec) / 1000.0f;
        return (*timer)->total_time + elapsed;
    }
    return (*timer)->total_time;
}

inline float sdkGetAverageTimerValue(StopWatchInterface **timer) {
    return sdkGetTimerValue(timer);
}

#endif // __HELPER_TIMER_H__

#endif // HIP_HELPER_H
```

## Skill prompt to create the file

> Read `reference/hip_helper.md` and write the C++ code block to
> `Common/hip_helper.h` in the current project directory.
