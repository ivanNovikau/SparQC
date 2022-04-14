#ifndef DEVICETABLE_PCS_H
#define DEVICETABLE_PCS_H


#include "device_data.cuh"


namespace PCS__
{
    // do not forget to zero the a_exch_ array before any gate operation;



/**
 * @brief PCS: Pauli X gate. There is data transfer between devices.
 * @param[in] t id of the target qubit (starts from 0);
 */
__global__ void PCC_gate_X(tkey t)
{
    // 


}


// /**
//  * @brief Place the KV pair \p kv into one of the subtables.
//  * This KV pair is treated by the whole group \p h_group.
//  * @param[in] kv Key-Value pair to place to the table;
//  * @param[in] h_group cooperative group, which works with the KV pair.
//  */ 
// __device__ __forceinline__ bool insert_within_group(
//     YCK key_to_insert,
//     YCV value_to_insert,
//     thread_block_tile<group_size_> h_group
// ){
//     KV kv_probe;
//     Subtable* st;
//     tkey hash_key;
//     uint32_t group_mask, id_leader, id_subtable;

//     bool flag_res = false;
//     auto id_loc_thr = h_group.thread_rank();

//     // if the group fails to insert the KV pair,
//     // the counter is incremented and changes the hash function.
//     for(
//         uint32_t counter_inc = 0; 
//         counter_inc < MIN_SUBTABLE_SIZE; 
//         counter_inc++
//     ){
//         if(counter_inc == (MIN_SUBTABLE_SIZE-1))
//             printf("!!! Max counter achieved !!!\n");

//         for(uint32_t id_g = 0; id_g < ngt_; id_g++)
//         {
//             flag_res = false;
//             id_subtable = id_g * group_size_ + id_loc_thr;

//             // printf("idT, counter, id-subtable: %u, %u, %u\n", threadIdx.x, counter_inc, id_subtable); 

//             st = &(table_d_.subtables_[id_subtable]);

//             // choose a position to place the KV pair there:
//             hash_key = table_helpers::get_hash(
//                 key_to_insert, 
//                 counter_inc, 
//                 id_subtable, 
//                 st->n_kv_
//             );
//             kv_probe = st->pairs_[hash_key];

//             // printf("idT, kv_probe.k: %u, %u\n", threadIdx.x, kv_probe.k);
//             // printf("idT, hash_key: %u, %u\n", threadIdx.x, hash_key);

//             // find an integer (common for all threads), where 
//             // i-th bit is set to 1 if the probe KV of the i-th thread is empty:
//             group_mask = h_group.ballot(kv_probe.k == KV::empty_key_);
    
            // group_mask = h_group.ballot(IS_ZERO(kv_probe.v));

//             // every thread with an empty seat competes to be chosen:
//             while(group_mask != 0)
//             {
//                 // get a thread with the smallest id, where the probe KV is empty: 
//                 id_leader = __ffs(group_mask) - 1;
//                 if(id_loc_thr == id_leader)
//                 {
//                     // while(atomicCAS(&mutex_d, 0, 1) != 0){}

//                     auto res_key = atomicCAS(
//                         &(st->pairs_[hash_key].k), 
//                         kv_probe.k, 
//                         key_to_insert
//                     );
//                     // printf("idT, key_to_insert, res_k, probe_k: %u, %u, %u, %u\n", 
//                     //     threadIdx.x, key_to_insert, res_key, kv_probe.k
//                     // );

//                     if(res_key == kv_probe.k){
//                         auto res_vr  = atomicCAS(
//                             (unsigned long long int*) 
//                             &(st->pairs_[hash_key].v.r), 
//                             __double_as_longlong(kv_probe.v.r),
//                             __double_as_longlong(value_to_insert.r)
//                         );
//                         if(res_vr  == kv_probe.v.r){
//                             auto res_vi  = atomicCAS(
//                                 (unsigned long long int*) 
//                                 &(st->pairs_[hash_key].v.i), 
//                                 __double_as_longlong(kv_probe.v.i),
//                                 __double_as_longlong(value_to_insert.i)
//                             );
//                             if(res_vi  == kv_probe.v.i) flag_res = true;
//                         }
//                     }

//                     // atomicExch(&mutex_d, 0);
//                 }

//                 if(h_group.any(flag_res == true)) // KV pair is placed !
//                     return true; 
//                 else // thread tries again to become the leader:
//                 {
//                     kv_probe = st->pairs_[hash_key];
//                     group_mask = h_group.ballot(kv_probe.k == KV::empty_key_);
//                 }
//             }
//         }
//     }

//     // TODO: handle error;
//     printf("!!! Error !!!: the insertion failed.\n");
//     return false; // the insertion failed !
// }


// __global__
// void insert_kv_pairs(tkey* keys_d, tvalue* values_d, uint32_t n_kv)
// {
//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group  = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);

//     // one group works with one kv pair to insert;
//     // within the group, every thread works with one subtable:
//     tkey key_to_insert;
//     for(
//         uint32_t id_kv = id_group; 
//         id_kv < n_kv; 
//         id_kv += n_groups 
//     ){
//         key_to_insert = keys_d[id_kv];
//         if(key_to_insert == KV::empty_key_)
//             continue;

//         // printf("idT, key-to-insert: %u, %u\n", threadIdx.x, key_to_insert);   

//         insert_within_group(
//             key_to_insert, values_d[id_kv], handle_group
//         );
//     }
// }


// __global__
// void resize_up_subtable(
//     KV* old_pairs,
//     tkey old_n_kv
// ){
//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);
//     auto id_loc_thr = handle_group.thread_rank();

//     KV one_kv;
//     for(
//         uint32_t id_kv = id_group; 
//         id_kv < old_n_kv; 
//         id_kv += n_groups 
//     ){
//         one_kv = old_pairs[id_kv];
//         if(one_kv.k == KV::empty_key_)
//             continue;

//         insert_within_group(
//             one_kv.k, one_kv.v, handle_group
//         );
//     }
// }









































// __global__
// void resize_up_subtable(
//     KV* old_pairs,
//     tkey old_n_kv,
//     uint32_t id_subtable_to_resize
// ){
//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);
//     auto id_loc_thr = handle_group.thread_rank();

//     // printf("idT, id_st: %u, %u\n", threadIdx.x, id_subtable_to_resize);

//     Subtable& new_subtable = table_d_.subtables_[id_subtable_to_resize];
//     KV one_kv;
//     tkey hash_key, id_kv;
//     tkey ngr_wi_sb = old_n_kv / group_size_ + 1;
//     for(tkey id_domain = id_group; id_domain < ngr_wi_sb; id_domain += n_groups )
//     {
//         id_kv = id_domain*group_size_ + id_loc_thr;
//         if(id_kv >= old_n_kv) break;

//         // printf("idT, id_kv: %u, %u\n", threadIdx.x, id_kv);

//         one_kv = old_pairs[id_kv];
//         if(one_kv.k != KV::empty_key_)
//         {
//             hash_key = table_helpers::get_hash_0(
//                 one_kv.k, id_subtable_to_resize, new_subtable.n_kv_
//             );

//             // check whether there is collision:
//             if(new_subtable.pairs_[hash_key].k != KV::empty_key_){
//                 printf("!!! ERROR !!!: collision during the scaling up.");
//                 assert(0);
//             }
//             new_subtable.pairs_[hash_key] = one_kv;
//         }
//     }
// }


// __global__
// void resize_down_subtable(
//     tbucket* old_subtable,
//     uint32_t old_nb,
//     uint32_t id_subtable_to_resize
// ){
//     tbucket* old_bucket; // bucket before the rescaling (bigger in size);
//     tbucket* new_bucket; // bucket after the rescaling (smaller in size);

//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);
//     auto id_loc_thr = handle_group.thread_rank();

//     tbucket* new_subtable = table_d_.subtables_[id_subtable_to_resize];
//     uint32_t new_nb = table_d_.subtable_sizes_[id_subtable_to_resize];

//     // --- copy the first half of the old subtable ---
//     for(
//         uint32_t id_bucket = id_group; 
//         id_bucket < new_nb; 
//         id_bucket += n_groups 
//     ){
//         old_bucket = old_subtable + id_bucket;
//         new_bucket = new_subtable + id_bucket;
//         for(
//             uint32_t id_kv = id_loc_thr; 
//             id_kv < BUCKET_SIZE; 
//             id_kv += group_size_
//         ) new_bucket->pairs_[id_kv] = old_bucket->pairs_[id_kv];
//     }

//     // --- copy the second hald of the old subtable --- 
//     // here, eviction of KV pairs to buckets of other subtables is possible:
//     bool flag_active = false;
//     KV kv_probe;
//     uint32_t group_mask, id_leader, id_init_table, seed_v;
//     KV kv_to_move;
//     for(
//         uint32_t id_bucket = id_group + new_nb; 
//         id_bucket < old_nb; 
//         id_bucket += n_groups 
//     ){
//         old_bucket = old_subtable + id_bucket;
//         for(
//             uint32_t id_kv = id_loc_thr; 
//             id_kv < BUCKET_SIZE; 
//             id_kv += group_size_
//         ){
//             flag_active = false;
//             kv_probe = old_bucket->pairs_[id_kv];
//             if(kv_probe.get_key() != kv_probe.empty_key_)
//                 flag_active = true;
            
//             group_mask = handle_group.ballot(flag_active);
//             while(group_mask != 0)
//             {
//                 id_leader = __ffs(group_mask) - 1;
//                 kv_to_move = handle_group.shfl(kv_probe, id_leader); 

//                 // evict KV pair to another subtable:
//                 seed_v = hashers::get_pair(kv_to_move.get_key());
//                 if(kv_to_move.get_key() & 1)
//                     id_init_table = hashers::get_table1_no(seed_v);
//                 else      
//                     id_init_table = hashers::get_table2_no(seed_v);
//                 insert_within_group(kv_to_move, id_init_table, handle_group);

//                 // repeat until all non-empty KV pairs from the bucket are removed;
//                 if(id_loc_thr == id_leader) flag_active = false;
//                 group_mask = handle_group.ballot(flag_active);
//             }
//         }
//     }
// }


// __device__ __forceinline__
// bool search_in_bucket(
//     const tkey& k, 
//     tvalue& v, 
//     tbucket* bb, 
//     thread_block_tile<group_size_> h_group
// ){
//     auto id_local_th = h_group.thread_rank();

//     KV kv_probe;
//     uint32_t group_mask, id_leader;
//     for(uint32_t id_g = 0; id_g < ngt_; id_g++)
//     {
//         kv_probe = bb->pairs_[id_g * group_size_ + id_local_th];
//         group_mask = h_group.ballot(kv_probe.get_key() == k);
//         if(group_mask != 0)
//         {
//             id_leader = __ffs(group_mask) - 1;
//             if(id_local_th == id_leader)
//                 v = kv_probe.get_value();
//             return true;
//         }
//     }
//     return false;
// }

// __global__
// void search_kv(tkey* keys_d, tvalue* values_d, uint32_t n_kv)
// {
//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);

//     tkey key_to_search;
//     uint32_t seed_v, id_table, nb, hash_key;
//     tbucket* bb;
//     bool flag_is = false;
//     for(
//         uint32_t id_kv = id_group; 
//         id_kv < n_kv; 
//         id_kv += n_groups 
//     ){
//         key_to_search = keys_d[id_kv];

//         // --- search in the first subtable ---
//         seed_v = hashers::get_pair(key_to_search);
//         if(key_to_search & 1)
//             id_table = hashers::get_table1_no(seed_v);
//         else        
//             id_table = hashers::get_table2_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         flag_is = search_in_bucket(key_to_search, values_d[id_kv], bb, handle_group);
//         if(handle_group.any(flag_is == true)) continue;

//         // --- search in the second subtable ---
//         if(key_to_search & 1)
//             id_table = hashers::get_table2_no(seed_v);
//         else        
//             id_table = hashers::get_table1_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         search_in_bucket(key_to_search, values_d[id_kv], bb, handle_group);
//     }
// }

// __device__ __forceinline__
// bool delete_in_bucket(
//     const tkey& k, 
//     uint32_t* nk_ptr,
//     tbucket* bb, 
//     thread_block_tile<group_size_> h_group
// ){
//     auto id_local_th = h_group.thread_rank();
//     KV kv_probe;
//     uint32_t group_mask, id_leader;
//     for(uint32_t id_g = 0; id_g < ngt_; id_g++)
//     {
//         kv_probe = bb->pairs_[id_g * group_size_ + id_local_th];
//         group_mask = h_group.ballot(kv_probe.get_key() == k);
//         if(group_mask != 0)
//         {
//             id_leader = __ffs(group_mask) - 1;
//             if(id_local_th == id_leader){
//                 atomicAdd(nk_ptr, 1);
//                 bb->pairs_[id_g * group_size_ + id_local_th] = KV();
//             }
                
//             return true;
//         }
//     }
//     return false;
// }

// __global__
// void delete_keys(tkey* keys_d, uint32_t* nk_ptr)
// {
//     uint32_t nk = nk_ptr[0];
//     nk_ptr[0] = 0;

//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);

//     tkey key_to_search;
//     uint32_t seed_v, id_table, nb, hash_key;
//     tbucket* bb;
//     bool flag_is = false;
//     for(
//         uint32_t id_kv = id_group; 
//         id_kv < nk; 
//         id_kv += n_groups 
//     ){
//         key_to_search = keys_d[id_kv];

//         // --- search in the first subtable ---
//         seed_v = hashers::get_pair(key_to_search);
//         if(key_to_search & 1)
//             id_table = hashers::get_table1_no(seed_v);
//         else        
//             id_table = hashers::get_table2_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         flag_is = delete_in_bucket(key_to_search, nk_ptr, bb, handle_group);
//         if(handle_group.any(flag_is == true)) continue;

//         // --- search in the second subtable ---
//         if(key_to_search & 1)
//             id_table = hashers::get_table2_no(seed_v);
//         else        
//             id_table = hashers::get_table1_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         flag_is = delete_in_bucket(key_to_search, nk_ptr, bb, handle_group);
//     }

// }


// __device__ __forceinline__
// bool replace_in_bucket(
//     const tkey& k, 
//     tvalue& v, 
//     int32_t* n_kv_ptr,
//     tbucket* bb, 
//     thread_block_tile<group_size_> h_group
// ){
//     auto id_local_th = h_group.thread_rank();

//     KV kv_probe;
//     uint32_t group_mask, id_leader;
//     for(uint32_t id_g = 0; id_g < ngt_; id_g++)
//     {
//         kv_probe = bb->pairs_[id_g * group_size_ + id_local_th];
//         group_mask = h_group.ballot(kv_probe.get_key() == k);
//         if(group_mask != 0)
//         {
//             id_leader = __ffs(group_mask) - 1;
//             if(id_local_th == id_leader)
//             {
//                 bb->pairs_[id_g * group_size_ + id_local_th].v = v;
//                 atomicAdd(n_kv_ptr, -1);
//             }
                
//             return true;
//         }
//     }
//     return false;
// }


// __global__
// void replace(tkey* keys_d, tvalue* values_d, int32_t* n_kv_ptr)
// {
//     int32_t n_kv = n_kv_ptr[0];
//     uint32_t n_groups = gridDim.x * blockDim.x / group_size_;
//     uint32_t id_group = (blockDim.x * blockIdx.x + threadIdx.x) / group_size_;
//     auto block_group = cg::this_thread_block(); 
//     auto handle_group = cg::tiled_partition<group_size_>(block_group);

//     tkey key_to_search;
//     uint32_t seed_v, id_table, nb, hash_key;
//     tbucket* bb;
//     bool flag_is = false;
//     for(
//         uint32_t id_kv = id_group; 
//         id_kv < n_kv; 
//         id_kv += n_groups 
//     ){
//         key_to_search = keys_d[id_kv];

//         // --- search in the first subtable ---
//         seed_v = hashers::get_pair(key_to_search);
//         if(key_to_search & 1)
//             id_table = hashers::get_table1_no(seed_v);
//         else        
//             id_table = hashers::get_table2_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         flag_is = replace_in_bucket(key_to_search, values_d[id_kv], n_kv_ptr, bb, handle_group);
//         if(handle_group.any(flag_is == true))
//         {
//             keys_d[id_kv] = KV::empty_key_;
//             continue;
//         }
            

//         // --- search in the second subtable ---
//         if(key_to_search & 1)
//             id_table = hashers::get_table2_no(seed_v);
//         else        
//             id_table = hashers::get_table1_no(seed_v);
//         nb = table_d_.subtable_sizes_[id_table];
//         hash_key = hashers::calculate_hash(key_to_search, id_table, nb);
//         bb = table_d_.subtables_[id_table] + hash_key;
//         flag_is = replace_in_bucket(key_to_search, values_d[id_kv], n_kv_ptr, bb, handle_group);
//         if(handle_group.any(flag_is == true))
//         {
//             keys_d[id_kv] = KV::empty_key_;
//         }
//     }
// }











} // end namespace PC__





#endif