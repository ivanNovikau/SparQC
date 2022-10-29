#ifndef TABLE_S_H
#define TABLE_S_H

#include "../include/data_t.h"

/** --------------------------------------------------------
 * --- HT: Hash Table, which sits on a single GPU device ---
 * ---------------------------------------------------------
 */
class Table__
{
public:
    KV* akvs_; // (HT1 and HT2): array of KVs.
    KV* a_exch_; // array to transfer KVs between GPUs.
    thash* ah_; // address book: (AH1 and AH2): array of all Hs: NHs are stored at the beginning of the array.

    uint32_t N_st_; // number of STs in HT.
    uint32_t capacity_st_; // capacity of one ST (number of positions to store KVs in a single ST).
    uint32_t Nb_; // number of buckets in a single ST.
    uint32_t bucket_size_; // number of positions in a single bucket;
    uint32_t capacity_; // maximum possible number of KVs in the HT.
    int32_t N_;        // number of NKVs in the HT.
    uint32_t counter_; // to count the current number of nonzero HTs;

    bool flag_gpus_; // are there exchanges between GPUs.
    uint32_t sh_[2]; // [0] - position in the current address book, [1] - position in the next address book.

    

    void reserve_on_device()
    {
        printf("\nReserve memory for the Table.\n");

        auto size_KVs  = sizeof(KV)    * capacity_;
        auto size_NHKs = 2*sizeof(thash) * capacity_;

        cudaMalloc((void**) &(akvs_), 2*size_KVs);
        if(flag_gpus_) 
        {
            printf("->create an array for the data exchange between GPUs.\n");
            cudaMalloc((void**) &(a_exch_), size_KVs);
        }
        cudaMalloc((void**) &(ah_),  size_NHKs); 
        checkCudaErrors(cudaGetLastError());
    }

    void get_statevector(std::unordered_map<tkey, tvalue>& vv, YCU counter)
    {
        KV* akv_host = new KV[capacity_];
        uint32_t sh = counter%2 == 0? 0: capacity_;
        cudaMemcpy(
            akv_host, 
            sh + akvs_, 
            sizeof(KV) * capacity_, // copy only HT1 or HT2 from akvs_;
            cudaMemcpyDeviceToHost
        );
        checkCudaErrors(cudaGetLastError());
        for(thash id_kv = 0; id_kv < capacity_; id_kv++)
        {
            // printf(" k, v = %ld, (%0.3f, %0.3f)\n", akv_host[id_kv].k - 1, akv_host[id_kv].v.r, akv_host[id_kv].v.i);

            tvalue& v1 = akv_host[id_kv].v;
            if( ComplexManager::is_negligible(v1) ) 
                continue;

            // printf(" k, v = %ld, (%0.3f, %0.3f)\n", akv_host[id_kv].k - 1, akv_host[id_kv].v.r, akv_host[id_kv].v.i);

            tkey& k1 = akv_host[id_kv].k;
            if(vv.find(k1) != vv.end())
                printf("Warning: the key [%lu] is already in the unordered statevector. Skip it.", k1);
            else
                vv.insert({k1 , v1});
        }
        delete [] akv_host;
    }

};

#endif