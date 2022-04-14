#include "../include/circuit.h"


using namespace std;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

int test_circuit(YCI argc, char *argv[]);

int main(int argc, char *argv[])
{
    int res;
    res = test_circuit(argc, argv);
    return res;
}

int test_circuit(YCI argc, char *argv[])
{
    hash_table_parameters hp;
    uint32_t nq;
    bool flag_only_circuit_parameters; // whether to investigate only basic circuit parameters.
    uint32_t id_arg = 0;

    cout << "\n-------------------------------------\n";
    cout << "--- SparQC ---\n";
    cout << "-------------------------------------\n";
    try
    {
        nq = 4;
        hp.N_devices = 1; 
        flag_only_circuit_parameters = false; 
        hp.N_subtables = 32;
        hp.coef_gpu_memory_reserve = 0.01;
        hp.log2_N_bucket_spots = 5;

        id_arg += 1;
        while(id_arg < (argc - 1))
        {
            if(YMIX::compare_strings(argv[id_arg], "-n_devices"))
            { 
                id_arg += 1;
                istringstream ss(argv[id_arg]);
                if(!(ss >> hp.N_devices))
                    throw string("command line argument: wrong format for the number of GPU devices.");
            }
            if(YMIX::compare_strings(argv[id_arg], "-f_only_init"))
            { 
                id_arg += 1;
                istringstream ss(argv[id_arg]);
                if(!(ss >> flag_only_circuit_parameters))
                    throw string("command line argument: wrong format of -f_only_init.");
            }
            if(YMIX::compare_strings(argv[id_arg], "-coef_gpu_mem"))
            { 
                id_arg += 1;
                istringstream ss(argv[id_arg]);
                if(!(ss >> hp.coef_gpu_memory_reserve))
                    throw string("command line argument: wrong format for -coef_gpu_mem.");
                if(hp.coef_gpu_memory_reserve < 0 || hp.coef_gpu_memory_reserve >= 1)
                    throw string("-coef_gpu_mem must be >= 0 and < 1.");
            }
            if(YMIX::compare_strings(argv[id_arg], "-N_subtables"))
            { 
                id_arg += 1;
                istringstream ss(argv[id_arg]);
                if(!(ss >> hp.N_subtables))
                    throw string("command line argument: wrong format for -N_subtables.");
            }
            if(YMIX::compare_strings(argv[id_arg], "-log2_n_bucket_spots"))
            { 
                id_arg += 1;
                istringstream ss(argv[id_arg]);
                if(!(ss >> hp.log2_N_bucket_spots))
                    throw string("command line argument: wrong format of -log2_n_bucket_spots.");
                if(hp.log2_N_bucket_spots > 5)
                    throw string("-log2_n_bucket_spots should be in the interval [0, 5].");
            }
            id_arg += 1;
        }
        cout << "* Number of devices: " << hp.N_devices << "\n";
        if(flag_only_circuit_parameters)
                std::cout << "* Only investigate basic circuit parameters.\n";
        cout << "* percentage of the GPU memory reserved for the HT arrays: " 
            << 1 - hp.coef_gpu_memory_reserve << "\n";

        // --- Create the quantum circuit ---
        Circuit__ oc(nq, hp);

        printf("--- Initial state ---\n");
        oc.print_full_statevector();

        if(!flag_only_circuit_parameters)
        {
            cout << "--- Circuit simulation ---" << endl;

            cout << "H[0]" << endl;
            oc.h(0);
            oc.print_full_statevector();

            cout << "Phase[0, 0.3]" << endl;
            oc.phase(0, 0.3);
            oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();


            // cout << "Z[0]" << endl;
            // oc.z(0);
            // oc.print_full_statevector();


            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();



            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // // cout << "Z[0]" << endl;
            // // oc.z(0);
            // // oc.print_full_statevector();

            // cout << "Y[0]" << endl;
            // oc.y(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();






            // oc.h(0);
            // oc.print_full_statevector();

            // oc.h(1);
            // oc.print_full_statevector();



            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[1]" << endl;
            // oc.h(1);
            // oc.print_full_statevector();



            // cout << "X[1]" << endl;
            // oc.x(1);
            // oc.print_full_statevector();

            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "H[1]" << endl;
            // oc.h(1);
            // oc.print_full_statevector();




            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "X[1]" << endl;
            // oc.x(1);
            // oc.print_full_statevector();

            // cout << "X[2]" << endl;
            // oc.x(2);
            // oc.print_full_statevector();

            // cout << "X[3]" << endl;
            // oc.x(3);
            // oc.print_full_statevector();

            // cout << "H[3]" << endl;
            // oc.h(3);
            // oc.print_full_statevector();

            // cout << "H[2]" << endl;
            // oc.h(2);
            // oc.print_full_statevector();

            // cout << "X[1]" << endl;
            // oc.x(1);
            // oc.print_full_statevector();

            // cout << "X[2]" << endl;
            // oc.x(2);
            // oc.print_full_statevector();

            // cout << "X[0]" << endl;
            // oc.x(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "H[0]" << endl;
            // oc.h(0);
            // oc.print_full_statevector();

            // cout << "H[2]" << endl;
            // oc.h(2);
            // oc.print_full_statevector();

            // cout << "H[3]" << endl;
            // oc.h(3);
            // oc.print_full_statevector();
        }
    }
    catch(const string& e)
    {
        std::cerr << "\n" << "Error: " << e << endl;
        return -1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "General error:\n" << e.what() << '\n';
        return -1;
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    std::cout << "Finished\n";

    return 0;
}