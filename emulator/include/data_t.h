#ifndef DATA_T_H
#define DATA_T_H

#include "sim_parameters_types.h"


// --------------------------------------------------------------
// --- Manager of complex values---
// --------------------------------------------------------------
class ComplexManager
{
public:
    static TComplex copy(YCCo from)
    {
        TComplex to;
        to.r = from.r;
        to.i = from.i;
        return to;
    }

    static inline 
    std::string get_line(
        YCCo v,
        YCB flag_scientific = false, 
        YCU prec = 3, 
        YCI init_line_w = -1
    ){
        int line_w = init_line_w;
        if(init_line_w == -1) 
            line_w = flag_scientific ? (prec + 10): (prec + 3);

        std::stringstream sstr;
        if(flag_scientific)
            sstr << std::scientific << std::setprecision(prec) <<
                std::setw(line_w+1) << v.r << " + " << 
                std::setw(line_w) << v.i << "j";
        else
            sstr << std::fixed      << std::setprecision(prec) << 
                std::setw(line_w+1) << v.r << " + " << 
                std::setw(line_w) << v.i << "j";
        return sstr.str();
    }

    __forceinline__ __host__ __device__ 
    static tfloat abs(YCCo v){return sqrt(v.r*v.r + v.i*v.i);}

    /**
     *  @brief Whether \p v is considered to be equal to zero.
     */
    __forceinline__ __host__ __device__ 
    static bool is_negligible(YCCo v)
    {
        bool flag_less = false;
        if(
            (std::abs(v.r) < ZERO_THRESHOLD) && 
            (std::abs(v.i) < ZERO_THRESHOLD)
        )
            flag_less = true;
        return flag_less;
    }

    // coef*(c1 + c2)
    __forceinline__ __host__ __device__
    static TComplex prod_add(YCQR coef, YCCo c1, YCCo c2)
    {
        return {
            coef * (c1.r + c2.r),
            coef * (c1.i + c2.i)
        };
    }

    // coef*(c1 - c2)
    __forceinline__ __host__ __device__
    static TComplex prod_sub(YCQR coef, YCCo c1, YCCo c2)
    {
        return {
            coef * (c1.r - c2.r),
            coef * (c1.i - c2.i)
        };
    }
};




    
// --------------------------------------------------------------
// --- Key-Value (KV) pair ---
// --------------------------------------------------------------
struct KV
{
public:
    tkey k;
    tvalue v;

    __host__ __device__
    KV()
    {
        set_kv(0, {0.0, 0.0});
    }

    __host__ __device__
    KV(tkey input_k, tvalue input_v)
    {
        set_kv(input_k, input_v);
    }

    __forceinline__ __host__ __device__
    void set_kv(tkey input_k, tvalue input_v)
    {
        k = input_k;
        v = input_v; 
    }

    static inline __host__
    std::string print_kv(
        const KV& kv,
        YCB flag_scientific = true, 
        YCU prec = 3, 
        YCI init_line_w = -1
    ){
        std::stringstream sstr;
        sstr << kv.k << ": \t" << ComplexManager::get_line(kv.v, flag_scientific, prec, init_line_w);
        return sstr.str();
    }
};

#endif

