# SparQC
Emulator of Sparse Quantum Circuits.
The emulated circuit is represented as a table, which in general consists on several _Hash Tables_ **(HTs)**.
Each HT sits on a single GPU.
Each HT consists on several _Subtables_ **(STs)**.
Each ST consists on several _buckets_.
Each bucket can store a certain number of statevector elements.

## Compilation
1. `mkdir emulator/build`
2. Choose the `CMAKE_BUILD_TYPE` inside `emulator/CMakeLists.txt` file: `Debug` or `Release`.
3. `mkdir emulator/bin_debug` or/and `mkdir emulator/bin_release`
4. `cd emulator/build`
5. `cmake ../`
6. `cmake --build ./`

## Run SparQC
To run the code:

`emulator/bin_[CMAKE_BUILD_TYPE]/sparqc [-n_devices] [-f_only_init] [-coef_gpu_mem] [-N_subtables] [-log2_n_bucket_spots]`

Here, the following parameters can be used:

`[-n_devices]`: `-n_devices N`: number of the GPUs used for the parallelization of the emulator. 
Be default, `N = 1`.

`[-f_only_init]`: `-f_only_init flag`: `flag = 0` stops the emulator after the circuit initialization (used to check the memory allocation and GPU parameters). Otherwise, the emulator computes the circuit.
Be default, `flag = 0`.

`[-coef_gpu_mem]`: `-coef_gpu_mem coef`: `coef < 1.0` is a part of the GPU memory, which is NOT used directly to store HT arrays.
Be default, `coef = 0.01`. 

`[-N_subtables]`: `-N_subtables N`: number of subtables in each HT.
Be default, `N = 32`. 
Depending on the resulting configuration (circuit size vs table size), the number of subtables can be changed by the emulator.

`[-log2_n_bucket_spots]`: `-log2_n_bucket_spots n`: log2 of the number of positions in a bucket.
Be default, `n = 5`, which means that there are `32` positions in a bucket to store `32` statevector elements there.




