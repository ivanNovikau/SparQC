#ifndef DEVICETABLE_FC_H
#define DEVICETABLE_FC_H

#include "device_data.cuh"

/**
 * Mapping: Full Circuit -> GHT = (
 *      a single HT (only one GPU),
 *      only one ST, 
 *      bucket_size = 1, 
 *      no cuckooing, 
 *      identity HF
 * ).
 */
namespace FC__
{
using namespace cooperative_groups;
namespace cg = cooperative_groups;

__device__ 
const auto& HF = table_helpers::hash_function_identity;


/**
 * @param[in] t target qubit.
 * @param[out] Nb  size of the statevector block.
 * @param[out] Nsb size of the statevector subblock.
 * @param[out] idx id of CUDA thread in the whole CUDA grid.
 */
__device__ __forceinline__
void prepare_gate(YCU t, uint64_t& Nb, uint64_t& Nsb, uint32_t& idx)
{
    Nb  = 1 << (t+1);
    Nsb = 1 << t;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
}


__global__ void set_to_zero_quantum_state() // set zero quantum state
{
    thash hk;
    HF(0, hk);
    table_d_.akvs_[hk] = {0, {1.0, 0.0}};
    table_d_.ah_[0] = hk;
    table_d_.ahi_[hk] = 0;
    table_d_.Ns_nkv_[0] = 1;
    table_d_.Ns_nkv_[1] = 1;
    table_d_.sh_[0] = 0;
    table_d_.sh_[1] = table_d_.capacity_;
}


/**
 * @brief Find K and HK of the element defined by \p id_thread and of its CE.
 * @param[in] id_thread id of CUDA thread in the whole CUDA grid.
 * @param[in] Nb   size of the statevector block.
 * @param[in] Nsb  size of the statevector subblock.
 * @param[in]  sh    to choose HT1 or HT2.
 * @param[out] hk_e HK of the element.
 * @param[out] hk_ce HK of the CE.
 * @return if true, skip the element.
 */
__device__ __forceinline__
bool find_elements(
    const uint32_t& id_thread, 
    YCK Nb, YCK Nsb, YCU sh,
    thash& hk_e, thash& hk_ce,
    tkey& ke, tkey& kce, bool& flag_within_upper_subblock
){
    tkey id_within_statevector_block; 

    hk_e = table_d_.ah_[sh + id_thread]; 
    KV& kv = table_d_.akvs_[sh + hk_e];
    ke = kv.k;

    if(ke == -1)
    {
        printf("Error: negative key");
        return true;
    }

    // printf("Inv.: kv.k, kv.v.r: \t%ld, %0.3f\n", ke, kv.v.r);

    id_within_statevector_block = ke % Nb;
    flag_within_upper_subblock = (id_within_statevector_block < Nsb) ? true: false;

    // printf("id_within_statevector_block, flag_within_upper_subblock, Nsb: %ld, %d, %lu\n", 
    //     id_within_statevector_block,
    //     flag_within_upper_subblock,
    //     Nsb
    // );

    // find K of CE and its H:
    if(flag_within_upper_subblock)
    {
        kce = kv.k + Nsb;
        HF(kce, hk_ce);
    }
    else
    {
        kce = kv.k - Nsb;
        HF(kce, hk_ce);
        
        // if the element key from the upper subblock is nonzero than skip:
        if( !IS_ZERO(table_d_.akvs_[sh + hk_ce].v) )
            return true;
    }
    return false;
}


template<uint32_t sel_gate>
__global__ void gate_sq(uint32_t t)
{
    thash hk_e, hk_ce; 
    uint32_t idx; 
    thash *Nnh_curr, *Nnh_new;
    uint64_t Nb, Nsb;
    tkey ke, kce;
    bool fu;
    uint32_t sh     = table_d_.sh_[0];
    uint32_t sh_new = table_d_.sh_[1];
    
    prepare_gate(t, Nb, Nsb, idx);
    choose_number_NHs(sh, Nnh_curr, Nnh_new);
    for(auto id_thread = idx; id_thread < *Nnh_curr; id_thread += blockDim.x * gridDim.x)
    {   
        if(find_elements(id_thread, Nb, Nsb, sh, hk_e, hk_ce, ke, kce, fu))
        {
            // printf("HERE: hk_e: %d\n", hk_e);
            if(sel_gate < 10)
            {
                // for gates that does not create any superposition of states:
                table_d_.ah_[sh_new + id_thread] = hk_e;
                table_d_.ahi_[sh_new + hk_e] = id_thread;
            }
            continue;
        } 
        if(sel_gate == 0)      x_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce);
        else if(sel_gate == 1) y_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu);
        else if(sel_gate == 2) z_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu);
        else if(sel_gate == 10) h_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu, Nnh_new);
    }
}


template<uint32_t sel_gate>
__global__ void gate_sq_par(uint32_t t, tfloat par)
{
    thash hk_e, hk_ce; 
    uint32_t idx; 
    thash *Nnh_curr, *Nnh_new;
    uint64_t Nb, Nsb;
    tkey ke, kce;
    bool fu;
    uint32_t sh     = table_d_.sh_[0];
    uint32_t sh_new = table_d_.sh_[1];
    tvalue aa = {cos(par), sin(par)};

    prepare_gate(t, Nb, Nsb, idx);
    choose_number_NHs(sh, Nnh_curr, Nnh_new);
    for(auto id_thread = idx; id_thread < *Nnh_curr; id_thread += blockDim.x * gridDim.x)
    {   
        if(find_elements(id_thread, Nb, Nsb, sh, hk_e, hk_ce, ke, kce, fu))
        {
            table_d_.ah_[sh_new + id_thread] = hk_e;
            table_d_.ahi_[sh_new + hk_e] = id_thread;
            continue;
        } 
        if(sel_gate == 0)      phase_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu, aa);
        else if(sel_gate == 1) rz_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu, aa);
        else if(sel_gate == 10) rx_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu, aa, Nnh_new);
        else if(sel_gate == 11) ry_core(id_thread, sh, sh_new, hk_e, hk_ce, ke, kce, fu, aa, Nnh_new);
    }
}







} // end namespace FC__
#endif