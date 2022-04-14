#ifndef TABLEMANAGER_S_H
#define TABLEMANAGER_S_H

#include "mapping_GHT.cuh"

/**
 * -----------------------------------------------------------
 * -> GHT: Global Hash Table: the whole hash table, where the circuit is mapped to.
 * The GHT is separated into several hash tables (HTs), where each of them is stored on 
 * a single GPU device.
 * ---
 * -> HT: hash table: each HT is stored on a single GPU device.
 *                  HT[H] = B, where B[i] = KV and KV = (K, V).
 * HT is split into subtables (ST) of equal sizes.
 * ---
 * -> K: Key (type tkey): index of an element in the circuit statevector.
 * ---
 * -> V: Value (type tvalue): amplitude of a statevector element;
 * ---
 * -> KV: Key-Value pair in HT: KV = (K, V).
 * ---
 * -> B: Bucket: a set of 2^n spots to store 2^n KV pairs: B[i] = KV.
 * ---
 * -> ZKV: Zero Key-Value pair: ZKV = (K, V: |V|~0).
 * ---
 * -> NKV: Nonzero Key-Value pair: NKV = (K, V: |V|>0).
 * ---
 * -> H: Hash key (type thash): index (position) of a KV inside HT. 
 * ---
 * -> ZH: Zero Hash key: ZH = -1.
 * ---
 * -> NH: Nonzero Hash key (> -1): HT[NH] = NKV.
 * ---
 * -> ST: Subtable: a section within HT.
 * ---
 * -> HF: Hash Function: mapping K -> H.
 * ---
 * -> CE: Companion Element. 
 * During the action of a single-qubit gate, the target qubit
 * defines a Companion Element for each element from the circuit statevector.
 * The new value of the element is calculated via 
 * the initial values of the element itself and of its CE 
 * multiplied by coefficients defined by the 2x2 matrix of the single-qubit gate.
 */
class TableManager__{
public:
    static constexpr uint32_t n_blocks_  = N_BLOCKS;
    static constexpr uint32_t n_threads_ = N_THREADS;

    // Mapping Circuit -> GHT:
    std::shared_ptr<Mapping__> conf_;

    // maximum number of KVs to store in the GHT:
    uint64_t N_max_all_;

    // number of GPU devices:
    uint32_t N_devices_;

    // number of all positions in the GHT (can be less than N_max_all_):
    uint64_t full_capacity_;

    // actual number of KVs, which will be stored on a single device:
    uint32_t capacity_ht_;

    // number of buckets in a HT:
    uint32_t Nb_;

    // number of spots in a single bucket:
    uint32_t bucket_size_;

    // HTs:
    std::vector<std::shared_ptr<Table__>> tables_;

    // are there transfers between GPUs:
    bool flag_gpus_;

    // counter of gates that change initial position in the HT address book:
    uint32_t counter_;

public:
    /**
     * @param[in] N_max_all_in a potential maximum number of KVs to store in GHT;
     * \p N_max_all_in can be bigger than the maximum capacity of the GHT;
     * @param[in] N_devices a number of GPU devices used to store the GHT;
     */
    TableManager__(YCUL N_max_all_in, const hash_table_parameters& hp) : 
        N_devices_(hp.N_devices),
        N_max_all_(N_max_all_in),
        flag_gpus_((hp.N_devices > 1) ? true : false)
    {
        // number of STs in each HT:
        uint32_t N_st;

        // number of KVs in one ST:
        uint32_t N_kv_st;

        // max. number of KVs, which can be stored on a single device: 
        uint32_t N_max_kv_device;

        // maximum number of concurrent CUDA blocks:
        uint32_t N_max_conc_blocks;

        // find the maximum number of KVs, which can be stored in one HT:      
        tables_.reserve(N_devices_);
        device_analysis(N_max_kv_device, N_max_conc_blocks, hp);
        if((N_devices_ > 1) && (N_max_kv_device >= N_max_all_)) 
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "WARNING: " << N_devices_ << " GPU devices are used.\n";
            std::cout << "However, one device is enough here:\n";
            std::cout << "circuit size = " << N_max_all_ << "\n";
            std::cout << "one device can store " << N_max_kv_device << " elements.\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }

        if(N_max_kv_device >= N_max_all_) 
        {
            N_st = 1;
            bucket_size_ = 1;
            Nb_ = N_max_all_;
        }
        else
        {
            N_st = hp.N_subtables;
            bucket_size_ = 1 << hp.log2_N_bucket_spots;
            Nb_ = N_max_kv_device / (bucket_size_ * N_st);
        }
        N_kv_st = Nb_ * bucket_size_;
        capacity_ht_ = N_st * N_kv_st;
        
        // create HTs of the GHT:
        full_capacity_ = N_devices_ * capacity_ht_;
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            cudaSetDevice(id_device);

            tables_.push_back(std::make_shared<Table__>());
            auto& table = tables_[id_device];

            table->capacity_st_ = N_kv_st; 
            table->capacity_ = capacity_ht_;
            table->N_st_ = N_st;
            table->Nb_ = Nb_;
            table->bucket_size_ = bucket_size_;
            table->flag_gpus_ = flag_gpus_;
            table->reserve_on_device();
            copy_meta_to_device(table.get());
        }

        // Set the mapping of the Circuit to the GHT:
        uint32_t N_blocks;
        if(N_max_all_ == capacity_ht_)
        {
            // one CUDA thread works with one NH:
            N_blocks = min(
                uint32_t((N_max_all_ - 1)/N_THREADS + 1),
                N_max_conc_blocks
            );
            conf_ = std::make_shared<MappingFC__>(N_blocks, bucket_size_);
        }
        else if((N_max_all_ > capacity_ht_) && (N_devices_ == 1))
        {
            // flag_PC_ = true;
            // line_res_config = "PC";
        }
        else if((N_max_all_ == full_capacity_) && (N_devices_ > 1))
        {
            // FCS
        }
        else if((N_max_all_ > capacity_ht_) && (N_devices_ > 1))
        {
            // flag_PCS_ = true;
            // line_res_config = "PCS";
        }
        else
        {
            throw std::string("Unknown Circuit->GHT mapping.");
        }
            
        // print GHT parameters:
        printf("\n--- GHT parameters ---\n");
        printf("  Number of GPU devices (= number of HTs): %u\n", N_devices_);
        printf("  ---\n");
        printf("  statevector size = \t%lu\n", N_max_all_);
        printf("  GHT full size: \t%lu\n", full_capacity_);
        printf("  ---\n");
        printf("  Log2(N-max-KVs-per-device), N-max-KVs-per-device: < %0.3f, < %u)\n", 
            log2(N_max_kv_device), N_max_kv_device
        );
        printf("  ---\n");
        printf("  Number of STs in an HT: \t%u\n", N_st);
        printf("  Number of buckets in a ST: \t%u\n", Nb_);
        printf("  Number of KVs in a bucket: \t%u\n", bucket_size_);
        printf("  Number of KVs in a ST: \t%u\n", N_kv_st);
        printf("  Number of KVs in an HT: \t%u\n", capacity_ht_);
        printf("  ---\n");
        printf("  statevector size / GHT size: \t\t%0.3e\n",    N_max_all_ / (1.*full_capacity_));
        printf("  statevector size / max HT size: \t%0.3e\n", N_max_all_ / (1.*N_max_kv_device));
        printf("  HT size / max HT size: \t\t%0.3e\n", capacity_ht_ / (1.*N_max_kv_device));
        printf("  ---\n");
        printf("  Resulting configuration: \t%s\n", conf_->line_conf_.c_str());
        printf("  N of threads per block: \t%u\n", N_THREADS);
        printf("  N of blocks:            \t%u\n", conf_->N_cuda_blocks_);
        printf("  Group size:             \t%u\n", bucket_size_);
        if(flag_gpus_)
            printf("  With data exchange between several GPUs.\n");
        else
            printf("  Without data exchange between GPUs.\n");
        printf("--------------------------------------------------------\n");

    }


    ~TableManager__()
    {
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            auto& table = tables_[id_device];
            cudaSetDevice(id_device);

            cudaFree(table->akvs_);
            cudaFree(table->ah_);
            cudaFree(table->ahi_);
            if(flag_gpus_) cudaFree(table->a_exch_);
        }
        checkCudaErrors(cudaGetLastError());
    }


    void device_analysis(uint32_t& N_max_kv_device, uint32_t& N_max_conc_blocks, const hash_table_parameters& hp)
    {
        double coef_GB = 1024.*1024.*1024.;
        double coef_kB = 1024.;

        // check whether we have enough devices:
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        if(nDevices < N_devices_)
        {
            std::stringstream sstr("");
            sstr << "There are not enough GPU devices:\n";
            sstr << "Requested N of devices: " << N_devices_ << "\n";
            sstr << "Available N of devices: " << nDevices << "\n";
            throw sstr.str();
        }

        // Assume that all devices have the same properties:
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // memory available to store KVs:
        auto avail_size = (1 - hp.coef_gpu_memory_reserve) * prop.totalGlobalMem;

        // size of one KV (including extra arrays): akvs_ (HT1 + HT2), a_exch_;
        auto one_kv = flag_gpus_ ? 3*sizeof(KV): 2*sizeof(KV);
        one_kv += 4*sizeof(thash); // + ah_ (AH1 + AH2) + ahi_;

        // maximum number of KVs to keep on a single GPU device:
        N_max_kv_device = avail_size/one_kv;
        if((N_max_kv_device * one_kv) > prop.totalGlobalMem)
            throw std::string("Error in the definition of the maximum possible number of KVs in HT.");

        // set the number of maximum concurrent blocks:
        N_max_conc_blocks = prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
        
        printf("\n--- GPU device properties ---\n");
        printf("  Device name: \t%s\n", prop.name);
        printf("  Compute capability: \t%d.%d\n", prop.major, prop.minor);
        printf("  Compute mode: \t%d\n", prop.computeMode);
        printf("  ---\n");
        printf("  Is on a multi-GPU board: \t%d\n", prop.isMultiGpuBoard);
        printf("  ---\n");
        printf("  N of SMs: \t\t%d\n", prop.multiProcessorCount);
        printf("  N of blocks per SM: \t%d\n", prop.maxBlocksPerMultiProcessor);
        printf("  max possible N of blocks per GPU: \t%d\n", N_max_conc_blocks);
        printf("  N of concurrent kernels: \t%d\n", prop.concurrentKernels);
        printf("  N of asynchronous engines: \t%d\n", prop.asyncEngineCount);
        printf("  ---\n");
        printf("  Max. number of threads per block: \t\t%u\n", prop.maxThreadsPerBlock);
        printf("  Max. number of blocks in x-direction: \t%d\n", prop.maxGridSize[0]);
        printf("  ---\n");
        printf("  Shared Memory per Block (kB): \t%0.3f\n", prop.sharedMemPerBlock/coef_kB);
        printf("  Constant Memory (kB): \t\t%0.3f\n", prop.totalConstMem/coef_kB);
        printf("  Global Memory (GB): \t\t%0.3f\n", prop.totalGlobalMem/coef_GB);
        printf("  ---\n");
        printf("  Memory that the max. number of KVs occupies (GB): \t%0.3f\n", 
            (N_max_kv_device * one_kv)/coef_GB
        );
        printf("-------------------------------------------\n");
    }


    static void copy_meta_to_device(Table__* table_ptr)
    {
        cudaMemcpyToSymbol(table_d_, table_ptr, sizeof(Table__));
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }


    void set_zero_quantum_state()
    {
        counter_ = 0;
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            cudaSetDevice(id_device);
            conf_->init();
            conf_->set_zero_quantum_state();
        }
    }


    template<uint32_t sel_gate>
    void gate_sq(YCU t)
    {
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            cudaSetDevice(id_device);
            if(sel_gate == 0)       conf_->x(t);
            else if(sel_gate == 1)  conf_->y(t);
            else if(sel_gate == 2)  conf_->z(t);
            else if(sel_gate == 10) conf_->h(t);
        }
        conf_->zero_arrays();
        ++counter_;
    }


    template<uint32_t sel_gate>
    void gate_sq_par(YCU t, YCQR par)
    {
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            cudaSetDevice(id_device);
            if(sel_gate == 0)       conf_->phase(t, par);
            else if(sel_gate == 1)  conf_->rz(t, par/2.);
            else if(sel_gate == 10) conf_->rx(t, par/2.);
            else if(sel_gate == 11) conf_->ry(t, par/2.);
        }
        conf_->zero_arrays();
        ++counter_;
    }


    void get_statevector(std::unordered_map<tkey, tvalue>& vv)
    {
        for(int id_device = 0; id_device < N_devices_; id_device++)
        {
            cudaSetDevice(id_device);
            tables_[id_device]->get_statevector(vv, counter_);
        }
    }


}; // end of the class TableManager__;





#endif