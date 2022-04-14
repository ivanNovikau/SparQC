#ifndef CIRCUIT_H
#define CIRCUIT_H


#include "../device/table_manager.cuh"
#include "QLib.h"


/**
 * Elements in the circuit statevector are enumerated from 0 up to (1<<nq - 1)
 */
class Circuit__{

protected:
    // number of qubits in the circuit:
    uint32_t nq_;

    // size of the circuit statevector:
    uint64_t N_;

    // table representation of the circuit:
    std::shared_ptr<TableManager__> tm_;

    // nonzero elements of the circuit statevector in the form of the hash table;
    // here, K starts from 0:
    std::unordered_map<tkey, tvalue> vv_;

public:

    /**
     * @param[in] nq number of qubits in the circuit;
     * @param[in] N_devices number of GPU devices to simulate the circuit;
     */
    Circuit__(YCU nq, const hash_table_parameters& hp) : 
        nq_(nq)
    {
        N_ = 1LL << nq_;
        tm_ = std::make_shared<TableManager__>(N_, hp);
        printf("\n--- Circuit created ---\n");
        printf("number of qubits, statevector size: %u, %lu\n", nq_, N_);
        printf("\n--- Circuit initialization ---\n");
        tm_->set_zero_quantum_state();
    }

    void x(YCU t)
    {
        tm_->gate_sq<0>(t);
    }

    void y(YCU t)
    {
        tm_->gate_sq<1>(t);
    }

    void z(YCU t)
    {
        tm_->gate_sq<2>(t);
    }

    void h(YCU t)
    {
        tm_->gate_sq<10>(t);
    }

    void phase(YCU t, YCQR par)
    {
        tm_->gate_sq_par<0>(t, par);
    }

    void rz(YCU t, YCQR par)
    {
        tm_->gate_sq_par<1>(t, par);
    }

    void rx(YCU t, YCQR par)
    {
        tm_->gate_sq_par<10>(t, par);
    }

    void ry(YCU t, YCQR par)
    {
        tm_->gate_sq_par<11>(t, par);
    }




    void print_raw_data()
    {
        form_unordered_statevector_DeviceToHost();
        print_unordered_statevector();
    }

    void print_full_statevector()
    {
        form_unordered_statevector_DeviceToHost();

        std::string str_wv;
        std::map<YVshv, tvalue> data;

        get_state_full({}, str_wv, data, {}, 3);
        std::cout << str_wv << std::endl;
    }

protected:
    /**
     * @brief Transfer unordered statevector with only nonzero KVs from GPUs to the host.
     */
    void form_unordered_statevector_DeviceToHost()
    {
        // std::cout << "* Transferring unordered statevector from Devices to Host..." << std::endl;
        if(vv_.empty())
        {
            // std::cout << "** Reserving memory for the unordered statevector..." << std::endl;
            auto N = tm_->full_capacity_;
            vv_.reserve(N);
        }
        else
        {
            // std::cout << "** Erasing elements from the unordered statevector..." << std::endl;
            vv_.erase(vv_.begin(), vv_.end());
        }
        tm_->get_statevector(vv_);
    }

    /**
     * @brief Print the unordered statevector with nonzero KVs.
     */
    void print_unordered_statevector()
    {
        KV kv;
        for(const auto& pair: vv_)
        {
            kv = {pair.first, pair.second};
            std::cout << KV::print_kv(kv) << "\n";
        }
    }

    // /**
    //  * @brief Form ordered set of bit-arrays.
    //  */
    // void get_state_full( 
    //     YCVU organize_state,
    //     YS str_wv, 
    //     YLVsh states, 
    //     YVCo ampls, 
    //     YCVsh state_to_choose,
    //     YCU ampl_prec
    // ){ 
    //     KV kv;
    //     bool flag_chosen;
    //     bool flag_to_choose; 

    //     // check whether it is necessary to choose a special state or not
    //     flag_to_choose = true;
    //     if(state_to_choose.empty()) flag_to_choose = false;

    //     ampls.clear();
    //     states.clear();
    //     for(const auto& pair: vv_)
    //     {
    //         kv = {pair.first, pair.second};

    //         // check whether the corresponding probability is not equal to zero
    //         if(!ComplexManager::is_negligible(kv.v))
    //         {
    //             flag_chosen = true;

    //             std::vector<short> one_state(nq_);
    //             YMATH::intToBinary(kv.k, one_state);

    //             // compare the computed state with the necessary pattern:
    //             if(flag_to_choose)
    //                 for(uint32_t id_qubit = 0; id_qubit < state_to_choose.size(); ++id_qubit)
    //                 {
    //                     if(state_to_choose[id_qubit] < 0)
    //                         continue;
    //                     if(state_to_choose[id_qubit] != one_state[id_qubit])
    //                     {
    //                         flag_chosen = false;
    //                         break;
    //                     }
    //                 }

    //             // save the state and its amplitude if the state corresponds to the imposed pattern:
    //             if(flag_chosen)
    //             {
    //                 ampls.push_back(kv.v);
    //                 states.push_back(one_state);
    //             }
    //         }
    //     }

    //     // form the resulting string:
    //     YMIX::getStrWavefunction(str_wv, ampls, states, organize_state, ampl_prec);
    // }


    /**
     * @brief Form ordered set of bit-arrays.
     * @param[out] data = {state: amplitude}
     */
    void get_state_full( 
        YCVU organize_state,
        YS str_wv, 
        std::map<YVshv, tvalue> data,
        YCVsh state_to_choose,
        YCU ampl_prec
    ){ 
        KV kv;
        bool flag_chosen;
        bool flag_to_choose; 

        // check whether it is necessary to choose a special state or not
        flag_to_choose = true;
        if(state_to_choose.empty()) flag_to_choose = false;

        data.clear();
        for(const auto& pair: vv_)
        {
            kv = {pair.first, pair.second};

            // check whether the corresponding probability is not equal to zero
            if(!ComplexManager::is_negligible(kv.v))
            {
                flag_chosen = true;

                std::vector<short> one_state(nq_);
                YMATH::intToBinary(kv.k, one_state);

                // compare the computed state with the necessary pattern:
                if(flag_to_choose)
                    for(uint32_t id_qubit = 0; id_qubit < state_to_choose.size(); ++id_qubit)
                    {
                        if(state_to_choose[id_qubit] < 0)
                            continue;
                        if(state_to_choose[id_qubit] != one_state[id_qubit])
                        {
                            flag_chosen = false;
                            break;
                        }
                    }

                // save the state and its amplitude if the state corresponds to the imposed pattern:
                if(flag_chosen)
                {
                    // data.insert({one_state, kv.v});
                    data[one_state] = kv.v;
                }
            }
        }

        // form the resulting string:
        YMIX::getStrWavefunction(str_wv, data, organize_state, ampl_prec);
    }







}; // end class Circuit








#endif