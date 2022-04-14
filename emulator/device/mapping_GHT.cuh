#ifndef MAPPING_GHT
#define MAPPING_GHT

#include "device_table_FC.cuh"


class Mapping__
{
public:
    // Number of CUDA blocks:
    uint32_t N_cuda_blocks_;

    // size of a CUDA cooperative group:
    uint32_t group_size_;

    // text notation of the configuration:
    std::string line_conf_;

    Mapping__(YCU N_blocks, YCU group_size) : 
        N_cuda_blocks_(N_blocks),
        group_size_(group_size)
    {
        line_conf_ = "None";
    }

    ~Mapping__(){}

    void init()
    {
        device_init_arrays<<<N_cuda_blocks_, N_THREADS>>>();
        checkCudaErrors(cudaGetLastError());
    }
    void zero_arrays(YCU sh)
    { 
        device_zero_arrays<<<N_cuda_blocks_, N_THREADS>>>(sh); 
    }

    virtual void set_zero_quantum_state() = 0;

    virtual void x(YCU t, YCU sh) = 0;
    virtual void y(YCU t, YCU sh) = 0;
    virtual void z(YCU t, YCU sh) = 0;
    virtual void h(YCU t, YCU sh) = 0;

    virtual void phase(YCU t, YCQR par, YCU sh) = 0;
    virtual void rz(YCU t, YCQR par, YCU sh) = 0;
    virtual void rx(YCU t, YCQR par, YCU sh) = 0;
    virtual void ry(YCU t, YCQR par, YCU sh) = 0;
};

/**
 * Full circuit on a single GPU.
 */
class MappingFC__ : public Mapping__
{
public:
    MappingFC__(YCU N_blocks, YCU group_size) : Mapping__(N_blocks, group_size)    
    {
        line_conf_ = "FC";
    }

    void set_zero_quantum_state(){ FC__::set_to_zero_quantum_state<<<1, group_size_>>>(); }

    void x(YCU t, YCU sh){ FC__::gate_sq<0><<<N_cuda_blocks_, N_THREADS>>>(t, sh); }
    void y(YCU t, YCU sh){ FC__::gate_sq<1><<<N_cuda_blocks_, N_THREADS>>>(t, sh); }
    void z(YCU t, YCU sh){ FC__::gate_sq<2><<<N_cuda_blocks_, N_THREADS>>>(t, sh); }
    void h(YCU t, YCU sh){ FC__::gate_sq<10><<<N_cuda_blocks_, N_THREADS>>>(t, sh); }

    void phase(YCU t, YCQR par, YCU sh){ FC__::gate_sq_par<0><<<N_cuda_blocks_, N_THREADS>>>(t, par, sh); }
    void rz(YCU t, YCQR par, YCU sh){    FC__::gate_sq_par<1><<<N_cuda_blocks_, N_THREADS>>>(t, par, sh); }
    void rx(YCU t, YCQR par, YCU sh){    FC__::gate_sq_par<10><<<N_cuda_blocks_, N_THREADS>>>(t, par, sh); }
    void ry(YCU t, YCQR par, YCU sh){    FC__::gate_sq_par<11><<<N_cuda_blocks_, N_THREADS>>>(t, par, sh); }
};




#endif