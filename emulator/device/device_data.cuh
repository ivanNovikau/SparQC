#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H

#include "table.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

/**
 * Indices of elements in HT starts from 0.
 * ---
 * -> FC: Full Circuit: full circuit is placed on a single GPU; 
 * circuit size <= GPU memory;
 * a single GPU, a single subtable, a single-size bucket;
 * HK is equivalent to K: HK == K.
 * HK and K are enumarated from 1 up to 1<<nq.
 * No cuckooing.
 * ---
 * -> PC: Part of the Circuit: part of the circuit is placed on a single GPU; 
 * circuit size > GPU memory;
 * single GPU, several subtables, bucket_size > 1.
 * With cuckooing.
 * ---
 * -> FCS: Full Circuit is calculated by Several GPUs;
 * several GPUs, one subtable on each GPU, single-size buckets.
 * No cuckooing. With data transfer between GPUs.
 * ---
 * -> PCS: Part of the Circuit is calculated by several GPUs;
 * several GPUs, several subtables on each GPU, buket_size > 1.
 * With cuckooing. With data transfer between GPUs.
 */


// HT on the device:
__constant__ Table__ table_d_;

__device__ 
const auto& IS_ZERO = ComplexManager::is_negligible;

__device__ 
const auto& C_PA = ComplexManager::prod_add;

__device__ 
const auto& C_PS = ComplexManager::prod_sub;


__global__ void device_init_arrays()
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    KV empty_kv = {-1, {0.0, 0.0}};
    for(auto id_thread = idx; id_thread < 2*table_d_.capacity_; id_thread += blockDim.x * gridDim.x)
    {
        table_d_.akvs_[id_thread] = empty_kv;
        table_d_.ah_[id_thread]   = -1;
        if(table_d_.flag_gpus_)
            table_d_.a_exch_[id_thread] = empty_kv;
    }
}


__global__ void device_zero_arrays()
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sh = table_d_.sh_[0];

    if(idx == 0) printf("--- zeroing arrays ---\n");
    if(idx == 0) printf("N-curr: %d\n", table_d_.N_);
    if(idx == 0) printf("N-new: %d\n",  table_d_.counter_);

    // for(auto id_thread = idx; id_thread < *Nnz_curr; id_thread += blockDim.x * gridDim.x)
    for(auto id_thread = idx; id_thread < table_d_.N_; id_thread += blockDim.x * gridDim.x)
    {
        auto& hk = table_d_.ah_[sh + id_thread]; 
        table_d_.akvs_[sh + hk] = KV{-1, {0.0, 0.0}};
        hk = -1;
    }
    if(idx == 0)
    {
        table_d_.N_ = table_d_.counter_;
        table_d_.counter_ = 0; // !!! bad for non-sup gates !!!

        if(table_d_.sh_[0] > 0)
        {
            table_d_.sh_[0] = 0;
            table_d_.sh_[1] = table_d_.capacity_;
        }
        else
        {
            table_d_.sh_[0] = table_d_.capacity_;
            table_d_.sh_[1] = 0;
        }
    }

    // // --- for testing ---
    // {
    //     // uint32_t sh_new = (sh > 0) ? 0: table_d_.capacity_;
    //     // for(auto id_thread = idx; id_thread < *Nnz_new; id_thread += blockDim.x * gridDim.x)
    //     // {
    //     //     auto& hk = table_d_.ah_[sh_new + id_thread];
    //     //     printf("id_thread = %d: hk:  %d\n", id_thread, hk);
    //     // }
    // }
    // if(idx == 0) printf("------\n");
}


__device__ __forceinline__
void update_ah_swap(YCU id_thread, YCU sh_new, YCH hk_e, YCH hk_ce, YCV vce)
{
    if(IS_ZERO(vce))
    {
        // printf("HERE 2: hk_ce: %d\n", hk_ce);
        table_d_.ah_[sh_new + id_thread] = hk_ce; 
    }
    else
    {
        // printf("HERE 3: hk_e: %d\n", hk_e);

        // nonzero CE is saved by other thread; 
        // save the element nonzero hash;
        table_d_.ah_[sh_new + id_thread] = hk_e; 
    }
} 


__device__ __forceinline__
void update_ah_diag(YCU id_thread, YCU sh_new, YCH hk_e)
{
    table_d_.ah_[sh_new + id_thread] = hk_e; 
}


__device__ __forceinline__
void update_ah_sup(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce)
{
    tvalue& ve_new  = table_d_.akvs_[sh_new + hk_e].v;
    tvalue& vce_new = table_d_.akvs_[sh_new + hk_ce].v;

    if(!IS_ZERO(ve_new) && !IS_ZERO(vce_new))
    {
        uint32_t pos = atomicAdd(&(table_d_.counter_), 2);
        table_d_.ah_[sh_new + pos] = hk_e;
        table_d_.ah_[sh_new + pos+1] = hk_ce;
    }
    else if(!IS_ZERO(ve_new))
    {
        uint32_t pos = atomicAdd(&(table_d_.counter_), 1);
        table_d_.ah_[sh_new + pos] = hk_e;
    }
    else if(!IS_ZERO(vce_new))
    {
        uint32_t pos = atomicAdd(&(table_d_.counter_), 1);
        table_d_.ah_[sh_new + pos] = hk_ce;
    }
}


__device__ __forceinline__
void x_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    // printf("sh: %u\n", sh);
    // printf("hk_e, hk_ce: \t%d, %d\n", hk_e, hk_ce);
    // printf("ke, k_ce: %ld, %ld\n", ke, kce);

    table_d_.akvs_[sh_new + hk_e]  = {ke,  vce};
    table_d_.akvs_[sh_new + hk_ce] = {kce, ve};
    update_ah_swap(id_thread, sh_new, hk_e, hk_ce, vce);
}


__device__ __forceinline__
void y_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {ke,  fu ? tvalue{vce.i, -vce.r}: tvalue{-vce.i, vce.r}};
    table_d_.akvs_[sh_new + hk_ce] = {kce, fu ? tvalue{-ve.i,   ve.r}: tvalue{  ve.i, -ve.r}};
    update_ah_swap(id_thread, sh_new, hk_e, hk_ce, vce);
}


__device__ __forceinline__
void z_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {ke,  fu ? ve: tvalue{-ve.r, -ve.i}};
    table_d_.akvs_[sh_new + hk_ce] = {kce, fu ? tvalue{-vce.r, -vce.i}: vce};
    update_ah_diag(id_thread, sh_new, hk_e);
}


__device__ __forceinline__
void h_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    tfloat f1 = 1./sqrt(2.);
    table_d_.akvs_[sh_new + hk_e]  = {ke,  fu ? C_PA(f1, ve, vce): C_PS(f1, vce, ve)};
    table_d_.akvs_[sh_new + hk_ce] = {kce, fu ? C_PS(f1, ve, vce): C_PA(f1, vce, ve)};
    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce);
}


__device__ __forceinline__
void phase_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {
        ke,  fu ? ve: tvalue{
            aa.r * ve.r - aa.i * ve.i, 
            aa.r * ve.i + aa.i * ve.r
        }
    };
    table_d_.akvs_[sh_new + hk_ce] = {
        kce, fu ? tvalue{
            aa.r * vce.r - aa.i * vce.i, 
            aa.r * vce.i + aa.i * vce.r
        }: vce
    };
    update_ah_diag(id_thread, sh_new, hk_e);
}


__device__ __forceinline__
void rz_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {
        ke,  fu ? 
            tvalue{
                aa.r * ve.r + aa.i * ve.i, 
                aa.r * ve.i - aa.i * ve.r
            }: 
            tvalue{
                aa.r * ve.r - aa.i * ve.i, 
                aa.r * ve.i + aa.i * ve.r
            }
    };
    table_d_.akvs_[sh_new + hk_ce] = {
        kce, fu ? 
            tvalue{
                aa.r * vce.r - aa.i * vce.i, 
                aa.r * vce.i + aa.i * vce.r
            }:            
            tvalue{
                aa.r * vce.r + aa.i * vce.i, 
                aa.r * vce.i - aa.i * vce.r
            }
    };
    update_ah_diag(id_thread, sh_new, hk_e);
}


__device__ __forceinline__
void rx_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {
        ke,   
        tvalue{
            aa.r * ve.r + aa.i * vce.i, 
            aa.r * ve.i - aa.i * vce.r
        }
    };
    table_d_.akvs_[sh_new + hk_ce] = {
        kce,
        tvalue{
            aa.r * vce.r + aa.i * ve.i, 
            aa.r * vce.i - aa.i * ve.r
        }
    };
    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce);
}


__device__ __forceinline__
void ry_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    table_d_.akvs_[sh_new + hk_e]  = {
        ke, fu ? 
            tvalue{
                aa.r*ve.r - aa.i*vce.r,
                aa.r*ve.i - aa.i*vce.i
            }:
            tvalue{
                aa.r*ve.r + aa.i*vce.r,
                aa.r*ve.i + aa.i*vce.i
            }
    };
    table_d_.akvs_[sh_new + hk_ce]  = {
        kce, fu ? 
            tvalue{
                aa.r*vce.r + aa.i*ve.r,
                aa.r*vce.i + aa.i*ve.i
            }:
            tvalue{
                aa.r*vce.r - aa.i*ve.r,
                aa.r*vce.i - aa.i*ve.i
            }
    };
    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce);
}








#endif