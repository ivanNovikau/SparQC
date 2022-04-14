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
        table_d_.ahi_[id_thread]  = -1;
        if(table_d_.flag_gpus_)
            table_d_.a_exch_[id_thread] = empty_kv;
    }
    if(idx == 0)
    {
        table_d_.Ns_nkv_[0] = 0;
        table_d_.Ns_nkv_[1] = 0;
    }
}


/**
 * @param[out] Nnz_curr current number of NKs in the HT.
 * @param[out] Nnz_new  new number of NKs in the HT.
 */
__device__ __forceinline__
void choose_number_NHs(YCU sh, thash*& Nnz_curr, thash*& Nnz_new)
{
    bool flag_curr_is_HT2 = (sh > 0) ? true: false;
    Nnz_curr = &(table_d_.Ns_nkv_[uint32_t(flag_curr_is_HT2)]);
    Nnz_new  = &(table_d_.Ns_nkv_[uint32_t(!flag_curr_is_HT2)]);
}


__global__ void device_zero_arrays()
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    thash *Nnz_curr, *Nnz_new;
    uint32_t sh = table_d_.sh_[0];
    choose_number_NHs(sh, Nnz_curr, Nnz_new);

    if(idx == 0) printf("--- zeroing arrays ---\n");
    if(idx == 0) printf("N-curr: %d\n", *Nnz_curr);
    if(idx == 0) printf("N-new: %d\n",  *Nnz_new);

    for(auto id_thread = idx; id_thread < *Nnz_curr; id_thread += blockDim.x * gridDim.x)
    {
        auto& hk = table_d_.ah_[sh + id_thread]; 
        table_d_.akvs_[sh + hk] = {-1, {0.0, 0.0}};
        table_d_.ahi_[sh + hk] = -1;
        hk = -1;
    }
    if(idx == 0)
    {
        *Nnz_curr = *Nnz_new;
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

    // --- for testing ---
    {
        // uint32_t sh_new = (sh > 0) ? 0: table_d_.capacity_;
        // for(auto id_thread = idx; id_thread < *Nnz_new; id_thread += blockDim.x * gridDim.x)
        // {
        //     auto& hk = table_d_.ah_[sh_new + id_thread];
        //     printf("id_thread = %d: hk:  %d\n", id_thread, hk);
        //     printf("id_thread = %d: ahi: %d\n", id_thread, table_d_.ahi_[sh_new + hk]);
        // }
    }
    if(idx == 0) printf("------\n");
}


__device__ __forceinline__
void update_ah_swap(YCU id_thread, YCU sh_new, YCH hk_e, YCH hk_ce, YCV vce)
{
    if(IS_ZERO(vce))
    {
        // printf("HERE 2: hk_ce: %d\n", hk_ce);

        // the element nonzero value changed its position inside the HT;
        table_d_.ah_[sh_new + id_thread] = hk_ce; 
        table_d_.ahi_[sh_new + hk_ce] = id_thread;
    }
    else
    {
        // printf("HERE 3: hk_e: %d\n", hk_e);

        // nonzero CE is saved by other thread; 
        // save the element nonzero hash;
        table_d_.ah_[sh_new + id_thread] = hk_e; 
        table_d_.ahi_[sh_new + hk_e] = id_thread;
    }
} 


__device__ __forceinline__
void update_ah_diag(YCU id_thread, YCU sh_new, YCH hk_e)
{
    table_d_.ah_[sh_new + id_thread] = hk_e; 
    table_d_.ahi_[sh_new + hk_e] = id_thread;
}


__device__ __forceinline__
void update_ah_sup(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCV ve, YCV vce, thash*& Nnh_new)
{
    tvalue& ve_new  = table_d_.akvs_[sh_new + hk_e].v;
    tvalue& vce_new = table_d_.akvs_[sh_new + hk_ce].v;

    // printf("ve, vce; ve_n, vce_n: %0.3f, %0.3f, %0.3f, %0.3f\n", ve.r, vce.r, ve_new.r, vce_new.r);
    
    // --> At least one resulting KV > 0.
    if(
        (!IS_ZERO(ve_new)&&!IS_ZERO(vce_new)) && 
        (!IS_ZERO(ve)&&!IS_ZERO(vce))
    )
    {
        printf("id_thread = %d: the same N\n", id_thread);

        table_d_.ah_[sh_new + id_thread] = hk_e;
        table_d_.ah_[sh_new + table_d_.ahi_[sh + hk_ce]] = hk_ce;
        table_d_.ahi_[sh_new + hk_e]  = id_thread;
        table_d_.ahi_[sh_new + hk_ce] = table_d_.ahi_[sh + hk_ce];
    }
    else if(
        (!IS_ZERO(ve_new)&&!IS_ZERO(vce_new)) && IS_ZERO(vce)
    ){
        printf("id_thread = %d: N + 1\n", id_thread);

        thash id_new_nh = atomicAdd(Nnh_new, 1);
        table_d_.ah_[sh_new + id_thread] = hk_e;
        table_d_.ah_[sh_new + id_new_nh] = hk_ce;
        table_d_.ahi_[sh_new + hk_e]  = id_thread;
        table_d_.ahi_[sh_new + hk_ce] = id_new_nh;
    }
    else if(
        IS_ZERO(ve_new) && !IS_ZERO(vce)
    ){
        printf("id_thread = %d: ve: N - 1\n", id_thread);

        atomicAdd(Nnh_new, -1);
        auto pos = min(id_thread, table_d_.ahi_[sh + hk_ce]);
        table_d_.ah_[sh_new + pos] = hk_ce;
        table_d_.ahi_[sh_new + hk_ce] = pos;
    }
    else if(
        IS_ZERO(vce_new) && !IS_ZERO(vce)
    ){
        printf("id_thread = %d: vce: N - 1\n", id_thread);

        atomicAdd(Nnh_new, -1);
        auto pos = min(id_thread, table_d_.ahi_[sh + hk_ce]);

        // printf("id_thread = %d: pos = %u\n", id_thread, pos);
        // printf("id_thread = %d: ahi = %d\n", id_thread, table_d_.ahi_[sh + hk_ce]);

        table_d_.ah_[sh_new + pos] = hk_e;
        table_d_.ahi_[sh_new + hk_e] = pos;
    }
    // printf("ve-new.r, vce-new.r: %0.3f, %0.3f\n", ve_new.r, vce_new.r);
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
void h_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, thash*& Nnh_new)
{
    tvalue& ve  = table_d_.akvs_[sh + hk_e].v;
    tvalue& vce = table_d_.akvs_[sh + hk_ce].v;

    tfloat f1 = 1./sqrt(2.);
    table_d_.akvs_[sh_new + hk_e]  = {ke,  fu ? C_PA(f1, ve, vce): C_PS(f1, vce, ve)};
    table_d_.akvs_[sh_new + hk_ce] = {kce, fu ? C_PS(f1, ve, vce): C_PA(f1, vce, ve)};

    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce, ve, vce, Nnh_new);
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
void rx_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa, thash*& Nnh_new)
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
    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce, ve, vce, Nnh_new);
}


__device__ __forceinline__
void ry_core(YCU id_thread, YCU sh, YCU sh_new, YCH hk_e, YCH hk_ce, YCK ke, YCK kce, YCB fu, YCV aa, thash*& Nnh_new)
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
    update_ah_sup(id_thread, sh, sh_new, hk_e, hk_ce, ve, vce, Nnh_new);
}








#endif