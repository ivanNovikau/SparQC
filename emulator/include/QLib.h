#ifndef QLIB_H
#define QLIB_H

#include "data_t.h"


// ------------------------------------------
// --- Type synonyms --- 
// ------------------------------------------

#define YCCM const std::shared_ptr<const YMATH::YMatrix>

// #define YQCP QCircuit*
// #define YSQ  std::shared_ptr<QCircuit>
// #define YSG  std::shared_ptr<Gate__>
// #define YSB  std::shared_ptr<Box__>
// #define YSM  std::shared_ptr<YMATH::YMatrix>
// #define YSCQ std::shared_ptr<const QCircuit>
// #define YCCQ const std::shared_ptr<const QCircuit>
// #define YCCB const std::shared_ptr<const Box__>
// #define YPQC YSQ




#define LOG_INDENT "    "
#define WHITESPACE " \n\r\t\f\v"

#define YSIZE_CHAR_ARRAY 100000

#define COMMENT_ORACLE    "//"
#define FORMAT_ORACLETOOL ".oracle"s 
#define FORMAT_PROFILE    ".condR_profile"s
#define FORMAT_CIRCUIT    ".circuit"s
#define FORMAT_ANGLES     ".angles"s
#define FORMAT_LOG        ".log"s
#define FORMAT_QSP        ".qsp"s
#define FORMAT_INIT       ".init_state"s
#define ENDING_FORMAT_OUTPUT "_OUTPUT.hdf5"s
#define ENDING_FORMAT_RESTART "_RESTART.hdf5"s

// ------------------------------------------
// --- Structure with an initial state --- 
// ------------------------------------------
struct INIT_STATE__
{
    bool flag_defined = false;
    long long b_ampl; // position of the first element in a state vector to set
    long long n_ampls; // number of elements to set
    YVQv ampl_vec_real;
    YVQv ampl_vec_imag;
};

// ------------------------------------------
// --- Structure with Global variables --- 
// ------------------------------------------
struct YGlobalVariables{

    std::string reg_whole_circuit = "the_whole_circuit";

    tfloat inv_sqrt2 = 1./sqrt(2);

    // matrix of the X-gate
    const ComplexMatrix2 mX = 
    {
        .real = {
            {0., 1.}, 
            {1., 0.}
        },
        .imag = {{0., 0.}, {0., 0.}}
    };

    // matrix of the Z-gate
    const ComplexMatrix2 mZ = 
    {
        .real = {
            {1.,  0.}, 
            {0., -1.}
        },
        .imag = {{0., 0.}, {0., 0.}}
    };

    // matrix of the Hadamard
    const ComplexMatrix2 mH = 
    {
        .real = {
            {inv_sqrt2,  inv_sqrt2}, 
            {inv_sqrt2, -inv_sqrt2}
        },
        .imag = {{0., 0.}, {0., 0.}}
    };

    ComplexMatrix2 mRy(YCQR a)
    {
        tfloat a2 = a/2.;
        ComplexMatrix2 res = {
            .real = {
                {cos(a2), -sin(a2)},
                {sin(a2),  cos(a2)}
            },
            .imag = {{0., 0.}, {0., 0.}}
        };
        return res;
    }

    ComplexMatrix2 mRz(YCQR a)
    {
        tfloat a2 = a/2.;
        ComplexMatrix2 res = {
            .real = {
                {cos(a2),      0.},
                {0.,      cos(a2)}
            },
            .imag = {
                {-sin(a2),      0.}, 
                {0.,       sin(a2)}
            }
        };
        return res;
    }

    ComplexMatrix2 mRc(YCQR az, YCQR ay)
    {
        tfloat az2 = az/2.;
        tfloat ay2 = ay/2.;

        // Ry(ay) * Rz(az)
        ComplexMatrix2 res = {
            .real = {
                {cos(az2)*cos(ay2), -cos(az2)*sin(ay2)},
                {cos(az2)*sin(ay2),  cos(az2)*cos(ay2)}
            },
            .imag = {
                {-sin(az2)*cos(ay2),  -sin(az2)*sin(ay2)}, 
                {-sin(az2)*sin(ay2),  sin(az2)*cos(ay2)}
            }
        };
        return res;
    }

    ComplexMatrix2 mPhase(YCQR a)
    {
        ComplexMatrix2 res = {
            .real = {
                {1., 0.},
                {0., cos(a)}
            },
            .imag = {
                {0., 0.}, 
                {0., sin(a)}
            }
        };
        return res;
    }


};

// ------------------------------------------
// --- Math functions --- 
// ------------------------------------------ 
namespace YMATH{
    /** Check if the variable \p x is zero.*/
    bool is_zero(YCQR x);

    /** 
    * @brief Convert the unsigned integer \p ui to an array of bits, \p binaryNum:
    * 2 -> {1, 0}; 
    * 6 -> {1, 1, 0} etc.
    * @param[out] binaryNum the resulting array of bits. 
    * REMARK: The size of the vector should be defined in advance.
    */
    void intToBinary(YCK ui, YVsh binaryNum);

    /**
     * @brief Convert the array of bits, \p bb, to the corresponding unsigned integer.
     * @param[out] bb vector of bits: bb[0] - the most significant bit. 
     * @return uint = bb[0]*2**(n - 1) + ... + bb[n-2]*2**1 + bb[n-1], where n is the vector size.
     */
    tkey binaryToInt(YCVsh bb);

    /**
     * @brief Two-dimensional matrix of a tfloat (which is usually double) type.
     * It is an object that provides some basic manipulations with a two-dimensional 
     * raw pointer.
     */
    class YMatrix{
        public:
            /**
             * @brief Create an empty matrix object whithout reserving any memory.
             */
            YMatrix();

            /**
             * @brief Create a zero matrix (\p Nrows, \p Ncols).
             * @param Nrows is a number of rows in a new matrix.
             * @param Ncols is a number of columns.
             * @param flag_zero if true, set elements to zero.
             */
            YMatrix(YCU Nrows, YCU Ncols, YCB flag_zero=true);

            /**
             * @brief Copy a matrix \p M. 
             */
            YMatrix(YCCM M);
            ~YMatrix();

            /**
             * @brief Create an empty matrix.
             */
            void create(YCU Nrows, YCU Ncols);

            /**
             * @brief Create a zero matrix.
             */
            void create_zero_matrix(YCU Nrows, YCU Ncols);

            /**
             * @brief Create an identity matrix.
             */
            void create_identity_matrix(YCU Nrows, YCU Ncols);

            void create_x10_matrix(YCU Nrows, YCU Ncols);

            /**
             * @brief Gives a raw pointer of the matrix.
             */
            tfloat** get_pointer();

            /**
             * @brief Create and return 1-D pointer to the matrix
             */
            tfloat* get_1d_pointer();

            /**
             * @brief Get a pointer to a transposed matrix
             */
            tfloat* get_1d_transposed_pointer();

            inline int get_nr(){return nr_;}
            inline int get_nc(){return nc_;}

            void set_squared_from_transposed_1d_matrix(int N, tfloat* M);

            inline
            tfloat& operator()(YCU id_row, YCU id_col)
            {
                if (id_row >= nr_)
                {
                    std::cerr << "\nError: id-row = " << id_row << ", while n-rows = " << nr_ << std::endl;
                    exit(-1);
                }
                if (id_col >= nc_)
                {
                    std::cerr << "\nError: id-column = " << id_col << ", while n-columns = " << nc_ << std::endl;
                    exit(-1);
                }
                return a_[id_row][id_col];
            }

            inline
            tfloat operator()(YCU id_row, YCU id_col) const
            {
                if (id_row >= nr_)
                {
                    std::cerr << "\nError: id-row = " << id_row << ", while n-rows = " << nr_ << std::endl;
                    exit(-1);
                }
                if (id_col >= nc_)
                {
                    std::cerr << "\nError: id-column = " << id_col << ", while n-columns = " << nc_ << std::endl;
                    exit(-1);
                }
                return a_[id_row][id_col];
            }

            /**
             * @brief Print the matrix with precision = \p prec.
             * @param prec precision of matrix elements.
             * @param flag_scientific if true, then print in the scientific notation.
             * @param wc width of every column.
             */
            void print(int prec=3, bool flag_scientific=false, int wc=2);

        protected:
            /**
             * @brief Reserve memory for a new matrix of a known size.
             * The function does not check whether the matrix has been already initialized.
             */
            void create_new();

            /**
             * @brief Free the memory occupied by the matrix.
             */
            void clear();

            /**
             * @brief Set elements to zeros.
             * The function does not check where the matrix has been already initialized.
             */
            void set_zeros();

        private:
            unsigned nr_, nc_;
            tfloat** a_ = nullptr;
            std::shared_ptr<tfloat[]> a_1d_ = nullptr;
            std::shared_ptr<tfloat[]> a_1d_transposed_ = nullptr;

    };

    ComplexMatrix2 inv_matrix2(const ComplexMatrix2& a);

    /**
     * @brief Create a vector with integers in the interval [start, end).
     */
    YVIv get_range(YCI start, YCI end);

    /**
     * @brief Whether a string \p str can be converted to a number.
     * @param str is a string to check;
     */
    bool is_number(YCS str);
}

// ------------------------------------------
// --- Mix functions --- 
// ------------------------------------------
namespace YMIX{
    void print_log(
        YCS line, 
        YCU n_indent = 0, 
        YCB flag_only_file=false, 
        YCB flag_new_line=true
    );
    void print_log_flush(YCS line, YCU n_indent=0);
    void print_log_err(YCS line);

    /** Timer*/
    struct YTimer{
        public:           
            void Start(){
                start_ = std::chrono::steady_clock::now();
            }
            void Stop(){
                end_ = std::chrono::steady_clock::now();
            }
            void StartPrint(YCS mess)
            {
                Start();
                print_log_flush(mess);
            }
            void StopPrint()
            {
                Stop();
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(3) 
                    << get_dur_s() << " s\n";
                print_log_flush(oss.str());
            }
            double get_dur(){
                std::chrono::duration<double> dur_seconds = end_ - start_;
                return 1000.*dur_seconds.count(); // in ms
            }
            double get_dur_s(){
                std::chrono::duration<double> dur_seconds = end_ - start_;
                return dur_seconds.count(); // in seconds
            }
            std::string get_dur_str_ms(){
                std::ostringstream ostr;
                ostr << std::scientific << std::setprecision(3) << get_dur() << " ms";
                return ostr.str();
            }
            std::string get_dur_str_s(){
                std::ostringstream ostr;
                ostr << std::scientific << std::setprecision(3) << get_dur()/1000. << " s";
                return ostr.str();
            }
        protected:
            std::chrono::time_point<std::chrono::steady_clock> start_;
            std::chrono::time_point<std::chrono::steady_clock> end_;
    };

    // void get_state_vector(const Qureg& qq, YCU nq, YVQ state_real, YVQ state_imag);

    // /** 
    // * @brief Form a string with the circuit statevector.
    // * @param[out] str_wv output string with the full statevector;
    // * @param[in]  ampls  input amplitudes of different \p states;
    // * @param[in]  states input states;
    // * @param[in]  organize_state input array that indicates how to output every state
    // *        e.g. if n_qubits = 4, and organize_state = [2,1,1], then
    // *        amplitude  |i3 i2> |i1> |i0>;
    // * @param[in] ampl_prec precision of amplitudes to output in \p str_wv;
    // */
    // void getStrWavefunction(
    //     YS str_wv, 
    //     YCVCo ampls, 
    //     YCLVsh states, 
    //     YCVU organize_state, 
    //     YCU ampl_prec = 3 
    // );

    
    /** 
    * @brief Form a string with the circuit statevector.
    * @param[out] str_wv output string with the full statevector;
    * @param[in]  data = {state: ampl};
    * @param[in]  organize_state input array that indicates how to output every state
    *        e.g. if n_qubits = 4, and organize_state = [2,1,1], then
    *        amplitude  |i3 i2> |i1> |i0>;
    * @param[in] ampl_prec precision of amplitudes to output in \p str_wv;
    */
    void getStrWavefunction(
        YS str_wv, 
        std::map<YVshv, tvalue> data,
        YCVU organize_state, 
        YCU ampl_prec = 3 
    );

    // /** Compute first \p n_states states starting from the low-priority qubits.
    //  * Returns only states with non-zero probability. 
    // * @param[in] qq circuit;
    // * @param[in] n_states a number of first low-priority to calculate.
    // * @param[in] organize_state input array that indicates how to output every state
    // *        e.g. if n_qubits = 4, and organize_state = [2,1,1], then
    // *        amplitude  |i3 i2> |i1> |i0>;
    // * @param[out] str_wv output string with a full state vector;
    // * @param[out] states output resulting states;
    // * @param[out] ampls  output amplitudes of different \p states;
    // * @param[in] state_to_choose array with bits that a state must have to be chosen:
    // *   Elements of this vector must be 0 or 1. 
    // *   -1 means that it can be either 0 or 1.
    // *   The first elements in the array correspond to the high-priority qubits.
    // *   The size of the array must take into account the ancillae.
    // * @param[in] prec input precision of amplitude to output in \p str_wv;
    // */
    // void Wavefunction_NonzeroProbability(
    //     const Qureg& qq, 
    //     YCU n_states,
    //     YCVU organize_state,
    //     YS str_wv, 
    //     YLVsh states, 
    //     YVCo ampls, 
    //     YCVsh state_to_choose = YVshv{},
    //     YCU prec = 3
    // );

    // /** Get only states with non-zero amplitudes 
    // * @param[in] states_init initial circuit states;
    // * @param[in] ampls_init  initial state amplitudes;
    // * @param[in] organize_state input array that indicates how to output every state
    // *        e.g. if n_qubits = 4, and organize_state = [2,1,1], then
    // *        amplitude  |i3 i2> |i1> |i0>;
    // * @param[in] ampl_prec input precision of amplitude to output in \p str_wv;
    // * @param[out] str_wv output string with states with only non-zero ampltides;
    // * @param[out] states output states with only non-zero ampltides;
    // * @param[out] ampls  output non-zero amplitudes of different \p states;
    // */
    // void getNonzeroWavefunction(
    //     YCLVsh states_init, 
    //     const std::vector<tvalue>& ampls_init, 
    //     const std::vector<unsigned>& organize_state, 
    //     std::string& str_wv, 
    //     YLVsh states, 
    //     std::vector<tvalue>& ampls, 
    //     const unsigned ampl_prec = 3 
    // );

    /** Get special states (but only with non-zero amplitudes).
    * @param[in] states_to_choose array with bits that \p states_init must have to be chosen:
    *   E.g. \p states_init = |x2 x1 x0>. 
    *   If states_to_choose = {0},   choose only states with x2 = 0.
    *   If states_to_choose = {0,1}, choose only states with x2 = 0, x1 = 1.
    *   If states_to_choose = {0,-1, 1}, choose only states with x2 = 0, x0 = 1 and any x1.
    * @param[in] state_init initial circuit states;
    * @param[in] ampls_init  initial state amplitudes;
    * @param[in] organize_state input array that indicates how to output every state
    *        e.g. if n_qubits = 4, and organize_state = [2,1,1], then
    *        amplitude  |i3 i2> |i1> |i0>;
    * @param[in] ampl_prec input precision of amplitude to output in \p str_wv;
    * @param[out] str_wv output string with chosen states;
    * @param[out] states chosen states;
    * @param[out] ampls  amplitudes of the chosen states \p states;
    */
    void getSpecialStates(
        YCVsh state_to_choose,
        YCLVsh states_init, 
        YVCo ampls_init, 
        YCVU organize_state, 
        YS str_wv, 
        YLVsh states, 
        YVCo ampls, 
        YCU ampl_prec
    );


    struct File
    {
        std::ofstream of; // stream connected to a file;
        
        /** Open a file.
         * @param[in] fileName name of the file;
         * @param[in] flagNew  is it a new file (if new, old data will be deleted);
         * */
        File(YCS fileName, bool flagNew=false)
        {
            if(fileName.empty())
            {
                std::cerr << "Error: File name is not set." << std::endl;
                exit(-1);
            }

            if(flagNew)
                of.open(fileName);
            else
                of.open(fileName, std::ios::app);
            if(!of)
            {
                std::cerr << "Error: It's impossible to open the file:\n" << fileName << std::endl;
                exit(-1);
            }
        }

        ~File()
        {
            if(of.is_open())
            {
                of.clear();
                of.close();
            }
        }

        template<class T>
        std::ostream& operator<<(const T& x)
        {
            of << x;
            return of;
        }
    };


    struct LogFile : File
    {
        static std::string name_global_;
        LogFile(bool flagNew=false) : File(name_global_, flagNew){}
    };

    /**
     * @brief Remove a comment from a line.
     * @param line line from where to remove a comment;
     * @return a new line;
     */
    std::string remove_comment(YCS line);

    // true if line1 == line2
    bool compare_strings(YCS line1, YCS line2);

    // true if line1 == line2 and in lines
    bool compare_strings(YCS line1, YCS line2, YCVS lines);

    // true if line1 in lines
    bool compare_strings(YCS line1, YCVS lines);

    template<class T>
    std::vector<T> conc_vectors(YCVT v1, YCVT v2)
    {
        std::vector<T> vv(v1);
        copy(v2.begin(), v2.end(), back_inserter(vv));

        return vv;
    }
    template<class T>
    std::vector<T> conc_vectors(YCVT v1, YCVT v2, YCVT v3)
    {
        std::vector<T> vv(v1);
        copy(v2.begin(), v2.end(), back_inserter(vv));
        copy(v3.begin(), v3.end(), back_inserter(vv));
        return vv;
    }
    template<class T>
    std::vector<T> conc_vectors(YCVT v1, YCVT v2, YCVT v3, YCVT v4)
    {
        std::vector<T> vv(v1);
        copy(v2.begin(), v2.end(), back_inserter(vv));
        copy(v3.begin(), v3.end(), back_inserter(vv));
        copy(v4.begin(), v4.end(), back_inserter(vv));
        return vv;
    }

    std::string get_line(std::vector<int> a);

    std::string ltrim(YCS s);
    std::string rtrim(YCS s);
    std::string trim(YCS s);

    void insert_indent(YS line_original, YCS line_indent);

    bool is_present(YCVS v, YCS e);
    bool is_present(YCVI v, YCI e);

    void get_array_from_list(
        YCLVsh v, 
        short* array_1d, 
        const unsigned long& nr, 
        const unsigned long& nc
    );

    uint32_t round_up_power_2(YCU x);
    uint64_t round_up_power_2(YCUL x);

    void get_current_date_time(YS line_date_time);
    struct H5File
    {
        /**
         * @brief Create and open an .hdf5 file with a name \p fname. 
         */
        void create(YCS fname);
        void close();

        inline void set_name(YCS fname){ name_ = fname; }

        /**
         * @brief Open an .hdf5 file with a name \p fname only to read it. 
         */
        void open_r();

        /**
         * @brief Open an .hdf5 file with a name \p fname to write-read it. 
         */
        void open_w();

        /**
         * @brief Add a group (folder) with a name \p gname to an already opened file.
         */
        void add_group(YCS gname);

        /**
         * @brief Add a dataset with a name \p dname, where a scalar \p v is to be written.
         * The dataset is put to a group \p gname.
         */
        template<class T>
        void add_scalar(const T& v, YCS dname, YCS gname)
        {
            if(!flag_opened) 
                throw "HDF5 File " + name_ + 
                    " is not opened to add a dataset " + dname + " to a group " + gname;
            // add_group(gname);

            H5::Group grp(f_->openGroup(gname));
            write(v, dname, grp);
        }

        template<class T>
        void add_vector(const std::vector<T>& v, YCS dname, YCS gname)
        {
            if(!flag_opened) 
                throw "HDF5 File " + name_ + 
                    " is not opened to add a dataset " + dname + " to a group " + gname;
            // add_group(gname);

            H5::Group grp(f_->openGroup(gname));
            write(v, dname, grp);
        }

        template<class T>
        void add_matrix(const std::list<std::vector<T>>& v, YCS dname, YCS gname)
        {
            if(!flag_opened) 
                throw "HDF5 File " + name_ + 
                    " is not opened to add a dataset " + dname + " to a group " + gname;
            // add_group(gname);

            T* array_1d;
            unsigned long nr, nc;

            nr = v.size();
            nc = v.back().size();
            array_1d = new T[nr*nc];

            YMIX::get_array_from_list(v, array_1d, nr, nc);

            H5::Group grp(f_->openGroup(gname));
            write(array_1d, nc, nr, dname, grp);

            delete [] array_1d;
        }

        template<class T>
        void read_scalar(T& v, YCS dname, YCS gname)
        {
            if(!flag_opened) 
                throw "HDF5 File " + name_ + 
                    " is not opened to read a dataset " + dname + " from a group " + gname;
            H5::Group grp(f_->openGroup(gname));
            read(v, dname, grp);
        }

        template<class T>
        void read_vector(std::vector<T>& v, YCS dname, YCS gname)
        {
            if(!flag_opened) 
                throw "HDF5 File " + name_ + 
                    " is not opened to add a dataset " + dname + " to a group " + gname;
            H5::Group grp(f_->openGroup(gname));
            read(v, dname, grp);
        }


        protected:
            inline void write(YCS v, YCS dname, H5::Group& grp)
            {
                auto dspace = H5::DataSpace(H5S_SCALAR);
                H5::StrType dtype(H5::PredType::C_S1, v.size()+1);
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write(v, dtype);
            }
            inline void write(YCI v, YCS dname, H5::Group& grp)
            {
                auto dspace = H5::DataSpace(H5S_SCALAR);
                auto dtype = H5::PredType::NATIVE_INT;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write((int*) &v, dtype);
            }
            inline void write(YCU v, YCS dname, H5::Group& grp)
            {
                auto dspace = H5::DataSpace(H5S_SCALAR);
                auto dtype = H5::PredType::NATIVE_UINT;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write((unsigned*) &v, dtype);
            }
            inline void write(const long unsigned int& v, YCS dname, H5::Group& grp)
            {
                auto dspace = H5::DataSpace(H5S_SCALAR);
                auto dtype = H5::PredType::NATIVE_ULONG;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write((long unsigned int*) &v, dtype);
            }
            inline void write(const double& v, YCS dname, H5::Group& grp)
            {
                auto dspace = H5::DataSpace(H5S_SCALAR);
                auto dtype = H5::PredType::NATIVE_DOUBLE;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write((int*) &v, dtype);
            }

            inline void write(YCVU v, YCS dname, H5::Group& grp)
            {
                hsize_t dims[] = {v.size()};
                H5::DataSpace dspace(1, dims);
                auto dtype = H5::PredType::NATIVE_UINT;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write(&v[0], dtype);
            }
            inline void write(const std::vector<double>& v, YCS dname, H5::Group& grp)
            {
                hsize_t dims[] = {v.size()};
                H5::DataSpace dspace(1, dims);
                auto dtype = H5::PredType::NATIVE_DOUBLE;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write(&v[0], dtype);
            }
            inline void write(const std::vector<TComplex>& v, YCS dname, H5::Group& grp)
            {
                hsize_t dims[] = {v.size()};
                H5::DataSpace dspace(1, dims);

                hid_t dtype = H5Tcreate(H5T_COMPOUND, sizeof(TComplex));
                H5Tinsert (dtype, "real", HOFFSET(TComplex,r), H5T_NATIVE_DOUBLE);
                H5Tinsert (dtype, "imag", HOFFSET(TComplex,i), H5T_NATIVE_DOUBLE);

                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write(&v[0], dtype);
            }

            inline void write(short* v, const unsigned long& nr, const unsigned long& nc, YCS dname, H5::Group& grp)
            {
                hsize_t dims[] = {nr, nc};
                H5::DataSpace dspace(2, dims);
                auto dtype = H5::PredType::NATIVE_SHORT;
                H5::DataSet dataset = grp.createDataSet(dname, dtype, dspace);
                dataset.write(v, dtype);
            }


            template<class T>
            inline void read(T& v, YCS dname, H5::Group& grp)
            {
                H5::DataSet dataset = grp.openDataSet(dname);
                H5::DataType dtype = dataset.getDataType();
                dataset.read(&v, dtype);
            }
            inline void read(YS v, YCS dname, H5::Group& grp)
            {
                H5::DataSet dataset = grp.openDataSet(dname);
                H5::DataType dtype = dataset.getDataType();
                v="";
                dataset.read(v, dtype);
            }
            template<class T>
            inline void read(std::vector<T>& v, YCS dname, H5::Group& grp)
            {
                H5::DataSet dataset = grp.openDataSet(dname);

                H5::DataSpace dataspace = dataset.getSpace();
                int rank = dataspace.getSimpleExtentNdims();
                hsize_t dims_out[rank];
                int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);

                unsigned long long N = 1;
                for(unsigned i_dim = 0; i_dim < rank; i_dim++)
                    N *= dims_out[i_dim];
                v = std::vector<T>(N);

                H5::DataType dtype = dataset.getDataType();

                dataset.read(&v[0], dtype, dataspace, dataspace);
            }

        protected:
            std::string name_;
            bool flag_opened;
            H5::H5File* f_;
    };
}


#endif