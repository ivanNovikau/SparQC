#ifndef DEVICETABLE_PC_H
#define DEVICETABLE_PC_H


#include "device_data.cuh"


namespace PC__
{


/**
 * @brief PC: Pauli X gate. No cuckooing.
 * @param[in] t id of the target qubit (starts from 0);
 */
__global__ void gate_X(uint32_t t)
{
    // !!! Attention !!!: tkey vs thash

    // uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
    // uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;

    // // consider 1 << (nq - 1) pairs of state vector elements;
    // if(id_group >= (table_d_.N_max_ >> 1)) 
    //     return; 

    // // uint32_t id_group_block = threadIdx.x / group_size_;
    // auto block_group  = cg::this_thread_block(); 
    // auto handle_group = cg::tiled_partition<group_size_>(block_group);

    // if(blockIdx.x == 0 && threadIdx.x == 0){
    //     printf("N_max, n_groups, COEF_SERIAL: %lu, %u, %u\n", table_d_.N_max_, n_groups, COEF_SERIAL);
    // }

    // tkey size_slab_half = 1LL << t;
    // tkey size_slab      = 1LL << (t+1);
    // if(blockIdx.x == 0 && threadIdx.x == 0){
    //     printf("size_block: %u\n", size_slab);
    // }

    // tkey for_shift_block = id_group/size_slab_half;
    // tkey id_element_u = for_shift_block * size_slab + id_group % size_slab_half;
    // tkey id_element_b = id_element_u + size_slab_half;


    // printf("idB, idT, idG, id_element_u, id_element_b: %u, %u, %u, %u, %u\n", 
    //     blockIdx.x, threadIdx.x, id_group, id_element_u, id_element_b
    // );




    // find keys id_element_u, id_element_b


    // when COEF_SERIAL = 0, then n_groups = N_max
    // for(tkey jb = 0; jb < 1<<(table_d_.Log2_N_max_ - t - 1); jb++)
    // {
    // }
    // // every group works with one key
    // for(
    //     uint32_t id_kv = id_group; 
    //     id_kv < table_d_.N_max_; 
    //     id_kv += n_groups 
    // ){


    // }
}


} // end namespace PC__





#endif