#ifndef SIMPARAMETERSTYPES_S_H
#define SIMPARAMETERSTYPES_S_H

#include <cmath>
#include <iostream>
#include <memory> // for smart pointers
#include <stdint.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>

#include <ctime>
#include <cstdlib>

#include <math.h>

#include <vector>
#include <list>
#include <map>
#include <unordered_map>

#include <iterator>
#include <algorithm>
#include <stdlib.h> 
#include <limits>
#include <chrono>
#include <stdarg.h>
#include <numeric>
#include <unistd.h>

#include <mpi.h>
#include <cuda_runtime_api.h>

#include "H5Cpp.h"

#include "../third_party/helper_cuda.h"

// --------------------------------------------
// --- SYSTEM PARAMETERS ---
// --------------------------------------------
#define YMPI  false
#define YCUDA true

#define ZERO_THRESHOLD 1e-10

#define N_BLOCKS    100      

// number of threads in one CUDA block: 
#define N_THREADS    16          

// minimum size of each subtable:
#define LOG2_MIN_SUBTABLE_SIZE 4



// The first 10 subtables have different hash function, then they repeat. 
// INSERT: one group works with one kv pair to insert; within the group, every thread works with one subtable.
//  If the thread cannot insert (find) the KV pair, it changes the counter in the subtable's hash function.


// --------------------------------------------
// --- TYPES ---
// --------------------------------------------
using YCB  = const bool&;

using YCsh  = const short&;
using YCVsh = const std::vector<short>&;
using YVsh  = std::vector<short>&;
using YVshv = std::vector<short>;

using YLVshv  = std::list<std::vector<short>>;
using YLVsh  = std::list<std::vector<short>>&;
using YCLVsh = const std::list<std::vector<short>>&;

using YCI  = const int32_t&;
using YCVI = const std::vector<int32_t>&;
using YVI  = std::vector<int32_t>&;
using YVIv = std::vector<int32_t>;

using YCU  = const uint32_t&;
using YCVU = const std::vector<uint32_t>&;
using YVU  = std::vector<uint32_t>&;
using YVUv = std::vector<uint32_t>;

using YCUL = const uint64_t&;

#define YCVT const std::vector<T>&

using YCS  = const std::string&;
using YS   = std::string&;
using YCVS = const std::vector<std::string>&;
using YVSv = std::vector<std::string>;

using tfloat = double;
struct TComplex
{
public:
    tfloat r; // real part;
    tfloat i; // imaginary part;
};
struct ComplexMatrix2
{
    tfloat real[2][2];
    tfloat imag[2][2];
};

using tkey     = int64_t; 
using thash    = int32_t; 
using tvalue   = TComplex;

using YCQR = const tfloat&;
using YCVQ = const std::vector<tfloat>&;
using YVQ  = std::vector<tfloat>&;
using YVQv = std::vector<tfloat>;

using YCCo  = const tvalue&;
using YCVCo = const std::vector<tvalue>&;
using YVCo  = std::vector<tvalue>&;
using YVCov = std::vector<tvalue>;

using YCK = const tkey&;
using YCH = const thash&;
using YCV = const tvalue&;

using YISS = std::istringstream&;

using YCVVI = const std::vector<std::vector<int>>&;
using YVVI  = std::vector<std::vector<int>>&;
using YVVIv = std::vector<std::vector<int>>;



struct hash_table_parameters{
    // number of GPU devices to simulate the quantum circuit:
    uint32_t N_devices;

    // percentage of the GPU memory not used for the storing of hash-table arrays:
    tfloat coef_gpu_memory_reserve;

    // number of subtables:
    uint32_t N_subtables;

    // log2 of the number of positions in a single bucket:
    uint32_t log2_N_bucket_spots;
};

// ------------------------------------------------------
// --- HELP FUNCTIONS ---
// ------------------------------------------------------
namespace table_helpers{

    __host__ __device__ __forceinline__ 
    void hash_function_identity(YCK k, thash& h)
    {
        // Does not work for big k (> sizeof(uint32_t)), 
        // since h is of a smaller-size type:
        h = k;
    }



    // Most of the hush functions are taken from https://github.com/zhuqiweigit/DyCuckoo
    #define PRIME_uint 294967291u

    __host__ __device__ __forceinline__ 
    thash hash1(tkey key) {
        key = ~key + (key << 15);
        key = key ^ (key >> 12);
        key = key + (key << 2);
        key = key ^ (key >> 4);
        key = key * 2057;
        key = key ^ (key >> 16);
        return (key);
    }

    __host__ __device__ __forceinline__
    thash hash2(tkey a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    __host__ __device__ __forceinline__  
    thash hash3(tkey sig) {
        return ((sig ^ 59064253) + 72355969) % PRIME_uint;
    }

    __host__ __device__ __forceinline__ 
    thash hash4(tkey a) {
        a = (a ^ 61) ^ (a >> 16);
        a = a + (a << 3);
        a = a ^ (a >> 4);
        a = a * 0x27d4eb2d;
        a = a ^ (a >> 15);
        return a;
    }

    __host__ __device__ __forceinline__ 
    thash hash5(tkey a) {
        a -= (a << 6);
        a ^= (a >> 17);
        a -= (a << 9);
        a ^= (a << 4);
        a -= (a << 3);
        a ^= (a << 10);
        a ^= (a >> 15);
        return a;
    }

    __host__ __device__ __forceinline__ 
    thash hash6(tkey key) {
        key = ~key + (key << 13);
        key = key ^ (key >> 10);
        key = key + (key << 3);
        key = key ^ (key >> 6);
        key = key * 1384;
        key = key ^ (key >> 14);
        return (key);
    }

    __host__ __device__ __forceinline__
    thash hash7(tkey a) {
        a = (a + 0x7ed55d16) + (a << 10);
        a = (a ^ 0xc761d23c) ^ (a >> 17);
        a = (a + 0x165667b1) + (a << 3);
        a = (a + 0xd3a1646c) ^ (a << 11);
        a = (a + 0xfd3046c5) + (a << 5);
        a = (a ^ 0xb54a4f09) ^ (a >> 14);
        return a;
    }

    __host__ __device__ __forceinline__
    thash hash8(tkey a) {
        a = (a + 0x7ec23d16) + (a << 8);
        a = (a ^ 0x9061d23c) ^ (a >> 15);
        a = (a + 0xbc5667b1) + (a << 5);
        a = (a + 0x89a1646c) ^ (a << 9);
        a = (a + 0x243046c5) + (a << 8);
        a = (a ^ 0xc24a4f09) ^ (a >> 10);
        return a;
    }

    __host__ __device__ __forceinline__ 
    thash hash9(tkey a) {
        a = (a ^ 51) ^ (a >> 14);
        a = a + (a << 2);
        a = a ^ (a >> 1);
        a = a * 0x27d3eb2d;
        a = a ^ (a >> 12);
        return a;
    }

    __host__ __device__ __forceinline__ 
    thash hash10(tkey a) {
        a -= (a << 4);
        a ^= (a >> 19);
        a -= (a << 10);
        a ^= (a << 2);
        a -= (a << 5);
        a ^= (a << 12);
        a ^= (a >> 10);
        return a;
    }

    __host__ __device__ __forceinline__ 
    thash hash_counter(tkey a, uint32_t counter) {
        a = (a ^ 32) ^ (counter >> 12);
        a = counter + (a << 2);
        a = a ^ (counter >> 1);
        a = a * 0x27c6eb2d;
        a = a ^ (counter >> 10);
        return a;
    }



    __host__ __device__ __forceinline__
    thash get_hash(tkey k, uint32_t counter, uint32_t id_subtable, uint64_t subtable_size){
        switch(id_subtable % 10){
            case 0:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash1(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 1:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash2(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 2:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash3(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 3:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash4(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 4:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash5(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 5:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash6(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 6:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash7(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 7:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash8(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 8:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash9(k) + counter*hash_counter(k, counter)) % subtable_size;
            case 9:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash10(k) + counter*hash_counter(k, counter)) % subtable_size;
        }
        return 0;
    }


    __host__ __device__ __forceinline__
    thash get_hash_0(tkey k, uint32_t id_subtable, uint64_t subtable_size){
        switch(id_subtable % 10){
            case 0:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash1(k)) % subtable_size;
            case 1:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash2(k)) % subtable_size;
            case 2:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash3(k)) % subtable_size;
            case 3:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash4(k)) % subtable_size;
            case 4:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash5(k)) % subtable_size;
            case 5:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash6(k)) % subtable_size;
            case 6:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash7(k)) % subtable_size;
            case 7:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash8(k)) % subtable_size;
            case 8:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash9(k)) % subtable_size;
            case 9:
                return ((subtable_size >> LOG2_MIN_SUBTABLE_SIZE) * hash10(k)) % subtable_size;
        }
        return 0;
    }

}


#endif