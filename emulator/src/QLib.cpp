#include "../include/QLib.h"
using namespace std;

std::string YMIX::LogFile::name_global_ = "output.log";


bool YMATH::is_zero(YCQR x)
{
    if(abs(x) < ZERO_THRESHOLD)
        return true;
    else
        return false;
}

void YMATH::intToBinary(YCK ui0, YVsh binaryNum)
{
    uint32_t i = 0;
    tkey ui = ui0;
    while (ui > 0) {
        binaryNum[i] = ui % 2;
        ui = ui / 2;
        i++;
    }

    // to guarantee that the high-priority bit is 
    // the mostleft, binaryNum[0]:
    std::reverse(binaryNum.begin(), binaryNum.end());
}

tkey YMATH::binaryToInt(YCVsh bb)
{
    int ii = 0;
    int count_b = bb.size();
    for(auto& b: bb)
    {
        --count_b;
        ii += b * int(pow(2, count_b));
    }
    return ii;
}

YMATH::YMatrix::YMatrix()
{
    nr_ = 0;
    nc_ = 0;
}
YMATH::YMatrix::YMatrix(YCU Nrows, YCU Ncols, YCB flag_zero)
    :nr_(Nrows), nc_(Ncols)
{
    create_new();
    if(flag_zero) set_zeros();
}
YMATH::YMatrix::YMatrix(YCCM oo)
{
    // copy the matrix oo to this object:
    nr_ = oo->nr_;
    nc_ = oo->nc_;
    create_new();

    for(int i = 0; i < nr_; ++i)
        for(int k = 0; k < nc_; ++k)
            a_[i][k] = oo->a_[i][k];
}

YMATH::YMatrix::~YMatrix()
{
    clear();
}

void YMATH::YMatrix::create_new()
{
    a_ = new tfloat*[nr_];
    for(int i = 0; i < nr_; ++i)
        a_[i] = new tfloat[nc_];
}

void YMATH::YMatrix::clear()
{
    if(nc_ != 0 || nr_ != 0)
    {
        // cout << "delete a matrix" << endl;
        for(unsigned i = 0; i < nr_; ++i) 
            delete [] a_[i];
        delete [] a_;

        nc_ = 0;
        nr_ = 0;

        a_1d_.reset();
    }
}

void YMATH::YMatrix::set_zeros()
{
    for(int i = 0; i < nr_; ++i)
        for(int k = 0; k < nc_; ++k)
            a_[i][k] = 0.0;
}

void YMATH::YMatrix::create(YCU Nrows, YCU Ncols)
{
    clear();
    nc_ = Ncols;
    nr_ = Nrows;
    create_new();
}

void YMATH::YMatrix::set_squared_from_transposed_1d_matrix(int N, tfloat* M)
{
    clear();
    nc_ = N;
    nr_ = N;
    create_new();
    
    for(int i = 0; i < nr_; ++i)
        for(int k = 0; k < nc_; ++k)
            a_[i][k] = M[k*N + i];
}

void YMATH::YMatrix::create_zero_matrix(YCU Nrows, YCU Ncols)
{
    clear();
    nc_ = Ncols;
    nr_ = Nrows;
    create_new();

    set_zeros();
}

void YMATH::YMatrix::create_identity_matrix(YCU Nrows, YCU Ncols)
{
    clear();
    nc_ = Ncols;
    nr_ = Nrows;
    create_new();

    for(int i = 0; i < nr_; ++i)
        for(int k = 0; k < nc_; ++k)
            if(i == k) a_[i][k] = 1.0;
            else       a_[i][k] = 0.0;
}

void YMATH::YMatrix::create_x10_matrix(YCU Nrows, YCU Ncols)
{
    clear();
    nc_ = Ncols;
    nr_ = Nrows;
    create_new();

    unsigned coef_10 = 0;
    for(unsigned i = 0; i < nr_; ++i)
    {
        coef_10 = 10*i;
        for(unsigned k = 0; k < nc_; ++k)
            a_[i][k] = coef_10 + k;
    }
}

tfloat** YMATH::YMatrix::get_pointer()
{
    return a_;
}

tfloat* YMATH::YMatrix::get_1d_pointer()
{
    if(!a_1d_ && nc_ > 0 && nr_ > 0)
    {
        a_1d_ = shared_ptr<tfloat[]>(new tfloat[nc_*nr_]);
        for(unsigned ir = 0; ir < nr_; ir++)
            for(unsigned ic = 0; ic < nc_; ic++)
                a_1d_[ir*nc_ + ic] = a_[ir][ic];
    }
    return a_1d_.get();
}

tfloat* YMATH::YMatrix::get_1d_transposed_pointer()
{
    if(!a_1d_transposed_ && nc_ > 0 && nr_ > 0)
    {
        a_1d_transposed_ = shared_ptr<tfloat[]>(new tfloat[nc_*nr_]);
        for(unsigned ic = 0; ic < nc_; ic++)
            for(unsigned ir = 0; ir < nr_; ir++)
                a_1d_transposed_[ic*nr_ + ir] = a_[ir][ic];
    }
    return a_1d_transposed_.get();
}

void YMATH::YMatrix::print(int prec, bool flag_scientific, int wc)
{
    for(unsigned i = 0; i < nr_; ++i)
    {
        for(unsigned k = 0; k < nc_; ++k)
            if(flag_scientific)
                std::cout << std::scientific << std::setprecision(prec) << std::setw(prec+wc) << a_[i][k] << " ";
            else
                std::cout << std::fixed << std::setprecision(prec) << std::setw(prec+wc+1) << a_[i][k] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

ComplexMatrix2 YMATH::inv_matrix2(const ComplexMatrix2& a)
{
    ComplexMatrix2 res;
    for(unsigned i = 0; i < 2; ++i)
        for(unsigned k = 0; k < 2; ++k)
        {
            res.real[i][k] =   a.real[k][i];
            res.imag[i][k] = - a.imag[k][i];
        }
    return res;
}

YVIv YMATH::get_range(YCI start, YCI end)
{
    int n = (end - start);
    YVIv res(n);
    std::iota(res.begin(), res.end(), start);
    return res;
}

bool YMATH::is_number(YCS str)
{
    for (char const &c : str) {
        if (std::isdigit(c) == 0) return false;
    }
    return true;
}



// void YMIX::get_state_vector(const Qureg& qq, YCU nq, YVQ state_real, YVQ state_imag)
// {
//     tvalue vv;
//     unsigned long long N = pow(2, nq);

//     state_real = YVQv(N);
//     state_imag = YVQv(N);
//     for(unsigned long long ii = 0; ii < N; ii++)
//     {
//         vv = getAmp(qq, ii);
//         state_real[ii] = vv.real;
//         state_imag[ii] = vv.imag;
//     }
// }

// void YMIX::getStrWavefunction(
//     YS str_wv, 
//     YCVCo ampls, 
//     YCLVsh states, 
//     YCVU organize_state, 
//     YCU ampl_prec
// ){ 
//     uint32_t n = states.front().size();
//     std::ostringstream oss;
//     oss << std::scientific << std::setprecision(ampl_prec);

//     str_wv = "";
//     uint32_t count_i = 0;
//     std::string str_state;
//     tvalue aa;
//     tfloat ar, ai;
//     uint32_t w_str  = 9 + ampl_prec;
//     for(auto& one_state:states){
//         str_state = "|";
//         if(organize_state.empty())
//             for(auto& one_qubit_state:one_state)
//                 str_state += std::to_string(one_qubit_state);
//         else{
//             uint32_t count_org = 0;
//             uint32_t prev_sum = 0;
//             for(uint32_t jj = 0; jj < n; ++jj){
//                 str_state += std::to_string(one_state[jj]);
//                 if(jj == prev_sum + organize_state[count_org]-1){
//                     prev_sum += organize_state[count_org];
//                     ++count_org;
//                     if(count_org < organize_state.size())
//                         str_state += ">|";
//                 }
//             }
//         }
//         str_state += ">";

//         oss.str(std::string());

//         aa = ampls.at(count_i);
//         ar = aa.r; 
//         ai = aa.i;

//         oss << std::setw(w_str) << ar;
//         oss << std::setw(w_str) << ai << "j";    
//         str_wv += oss.str() + "   " + str_state + "\n";

//         ++count_i;
//     }
// }



void YMIX::getStrWavefunction(
    YS str_wv, 
    std::map<YVshv, tvalue> data,
    YCVU organize_state, 
    YCU ampl_prec
){ 
    uint32_t n = data.begin()->first.size(); 
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(ampl_prec);

    str_wv = "";
    uint32_t count_i = 0;
    std::string str_state;
    tvalue aa;
    tfloat ar, ai;
    uint32_t w_str  = 9 + ampl_prec;
    for(auto const& [one_state, aa]: data)
    {
        str_state = "|";
        if(organize_state.empty())
            for(auto& one_qubit_state:one_state)
                str_state += std::to_string(one_qubit_state);
        else{
            uint32_t count_org = 0;
            uint32_t prev_sum = 0;
            for(uint32_t jj = 0; jj < n; ++jj){
                str_state += std::to_string(one_state[jj]);
                if(jj == prev_sum + organize_state[count_org]-1){
                    prev_sum += organize_state[count_org];
                    ++count_org;
                    if(count_org < organize_state.size())
                        str_state += ">|";
                }
            }
        }
        str_state += ">";

        oss.str(std::string());

        ar = aa.r; 
        ai = aa.i;

        oss << std::setw(w_str) << ar;
        oss << std::setw(w_str) << ai << "j";    
        str_wv += oss.str() + "   " + str_state + "\n";
    }
}



// void YMIX::Wavefunction(
//     const Qureg& qq, 
//     YS str_wv, 
//     YLVsh states, 
//     YVCo ampls, 
//     YCVU organize_state, 
//     YCU ampl_prec
// ){ 
//     const unsigned n = qq.numQubitsRepresented; // number of qubits;
//     unsigned N = pow(2, n);
//     tvalue amp;

//     // find states available for such a number of qubits and their amplitudes:
//     for(unsigned ii = 0; ii < N; ii++){
//         std::vector<short> one_state(n);
//         YMATH::intToBinary(ii, one_state);
//         states.push_back(one_state);
//         amp = getAmp(qq, ii);
//         ampls.push_back(amp);
//     }

//     // form the resulting string:
//     getStrWavefunction(str_wv, ampls, states, organize_state, ampl_prec);
// }

// void YMIX::Wavefunction_NonzeroProbability(
//     const Qureg& qq, 
//     YCU n_states,
//     YCVU organize_state,
//     YS str_wv, 
//     YLVsh states, 
//     YVCo ampls, 
//     YCVsh state_to_choose,
//     YCU ampl_prec
// ){ 
//     const unsigned n = qq.numQubitsRepresented; // number of qubits;
//     long long N = pow(2, n_states);
//     tvalue aa;
//     tfloat prob2;
//     bool flag_chosen;
//     bool flag_to_choose; 

//     // check whether it is necessary to choose a special state or not
//     flag_to_choose = true;
//     if(state_to_choose.empty()) flag_to_choose = false;

//     // find states available for such a number of qubits and their amplitudes:
//     ampls.clear();
//     states.clear();
//     for(unsigned id_state = 0; id_state < N; id_state++){

//         // calculate the amplitude of a state
//         aa = getAmp(qq, id_state);

//         // check whether the corresponding probability is not equal to zero
//         prob2 = pow(aa.real, 2) + pow(aa.imag, 2);
//         if(!YMATH::is_zero(prob2)){
//             flag_chosen = true;

//             vector<short> one_state(n);
//             YMATH::intToBinary(id_state, one_state);

//             // compare the computed state with a necessary pattern:
//             if(flag_to_choose)
//                 for(unsigned id_qubit = 0; id_qubit < state_to_choose.size(); ++id_qubit)
//                 {
//                     if(state_to_choose[id_qubit] < 0)
//                         continue;
//                     if(state_to_choose[id_qubit] != one_state[id_qubit])
//                     {
//                         flag_chosen = false;
//                         break;
//                     }
//                 }

//             // save the state and its amplitude if it corresponds to the pattern:
//             if(flag_chosen)
//             {
//                 ampls.push_back(aa);
//                 states.push_back(one_state);
//             }
//         }
//     }

//     // form the resulting string:
//     getStrWavefunction(str_wv, ampls, states, organize_state, ampl_prec);
// }

// void YMIX::getNonzeroWavefunction(
//     YCLVsh states_init, 
//     const std::vector<tvalue>& ampls_init, 
//     const std::vector<unsigned>& organize_state, 
//     std::string& str_wv,
//     YLVsh states, 
//     std::vector<tvalue>& ampls, 
//     const unsigned ampl_prec
// ){
//     tvalue aa;
//     tfloat prob2;
//     unsigned count_state = 0;
//     for(auto& one_state:states_init){
//         aa = ampls_init[count_state];
//         prob2 = pow(aa.real, 2) + pow(aa.imag, 2);
//         if(!YMATH::is_zero(prob2)){
//             ampls.push_back(aa);
//             states.push_back(one_state);
//         }
//         ++count_state;
//     } 

//     // form the resulting string:
//     getStrWavefunction(str_wv, ampls, states, organize_state, ampl_prec);
// }

// void YMIX::getSpecialStates(
//     YCVsh state_to_choose,
//     YCLVsh states_init, 
//     YVCo ampls_init, 
//     YCVU organize_state, 
//     YS str_wv, 
//     YLVsh states, 
//     YVCo ampls, 
//     YCU ampl_prec
// ){
//     tvalue aa;
//     bool flag_chosen;
//     unsigned count_state = 0;
//     for(auto& one_state:states_init){
//         flag_chosen = true;

//         for(unsigned i = 0; i < state_to_choose.size(); ++i)
//         {
//             if(state_to_choose[i] < 0)
//                 continue;
//             if(state_to_choose[i] != one_state[i])
//             {
//                 flag_chosen = false;
//                 break;
//             }
//         }

//         if(flag_chosen){
//             ampls.push_back(ampls_init[count_state]);
//             states.push_back(one_state);
//         }
//         ++count_state;
//     }

//     // form the resulting string:
//     getStrWavefunction(str_wv, ampls, states, organize_state, ampl_prec);
// }

string YMIX::remove_comment(YCS line)
{
    string new_line = line.substr(0, line.find(COMMENT_ORACLE, 0));
    return new_line;
}

bool YMIX::compare_strings(YCS line1, YCS line2)
{
    string new_line1(line1), new_line2(line2);
    transform(new_line1.begin(), new_line1.end(), new_line1.begin(), ::tolower);
    transform(new_line2.begin(), new_line2.end(), new_line2.begin(), ::tolower);

    if(new_line1.compare(new_line2) == 0)
        return true;
    else
        return false;
}

bool YMIX::compare_strings(YCS line1, YCS line2, YCVS lines)
{
    if(compare_strings(line1, line2))
        for(const auto& line_one: lines)
            if(compare_strings(line1, line_one))
                return true;
    return false;
}

bool YMIX::compare_strings(YCS line1, YCVS lines)
{
    for(const auto& line_one: lines)
        if(compare_strings(line1, line_one))
            return true;
    return false;
}



string YMIX::get_line(std::vector<int> a)
{
    ostringstream inf;
    for(auto const& a1: a) inf << a1 << "  ";
    return inf.str();
}

void YMIX::print_log(YCS line, YCU n_indent, YCB flag_only_file, YCB flag_new_line)
{
    string line_print = line;

    if(n_indent>0)
    {
        string str_indent = "";
        for(uint32_t i = 0; i < n_indent; i++) str_indent += LOG_INDENT;
        insert_indent(line_print, str_indent);
    }
    
    YMIX::LogFile cf;
    cf << line_print; 
    if(flag_new_line) cf << "" << endl;

    if(!flag_only_file)
    {
        cout << line_print;
        if(flag_new_line) cout << "" << endl;
    }
}
void YMIX::print_log_flush(YCS line, YCU n_indent)
{
    string line_print = line;
    if(n_indent>0)
    {
        string str_indent = "";
        for(uint32_t i = 0; i < n_indent; i++) str_indent += LOG_INDENT;
        insert_indent(line_print, str_indent);
    }
    
    YMIX::LogFile cf;
    cf   << line_print << flush; 
    cout << line_print << flush;
}
void YMIX::print_log_err(YCS line)
{
    throw line;
}

string YMIX::ltrim(YCS s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == string::npos) ? "" : s.substr(start);
}

string YMIX::rtrim(YCS s)
{
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == string::npos) ? "" : s.substr(0, end + 1);
}

string YMIX::trim(YCS s) {
    return rtrim(ltrim(s));
}

void YMIX::insert_indent(YS line_original, YCS line_indent)
{
    stringstream sstr(line_original);
    string one_line;

    getline(sstr, one_line);
    string line_res = line_indent + one_line;

    while(getline(sstr, one_line)) line_res += "\n" + line_indent + one_line;
    line_original = line_res;
}

bool YMIX::is_present(YCVS v, YCS e)
{
    if(find(v.begin(), v.end(), e) == v.end())
        return false;
    return true;
}
bool YMIX::is_present(YCVI v, YCI e)
{
    if(find(v.begin(), v.end(), e) == v.end())
        return false;
    return true;
}

void YMIX::get_current_date_time(YS line_date_time)
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%m-%d-%Y %H:%M:%S", timeinfo);
    line_date_time = string(buffer);
}

void YMIX::H5File::create(YCS fname)
{

    if(!fname.empty()) set_name(fname);
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_TRUNC);
    flag_opened = true;
}

void YMIX::H5File::open_r()
{
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_RDONLY);
    flag_opened = true;
}

void YMIX::H5File::open_w()
{
    f_ = new H5::H5File(name_.c_str(), H5F_ACC_RDWR);
    flag_opened = true;
}

void YMIX::H5File::close()
{
    delete f_;
    flag_opened = false;
}

void YMIX::H5File::add_group(YCS gname)
{
    H5::Group grp(f_->createGroup(gname));
}

void YMIX::get_array_from_list(
    YCLVsh v, 
    short* array_1d, 
    YCUL nr, 
    YCUL nc
){
    // !!! assume that every vector has the same amount of elements !!!
    uint64_t count_r = 0;
    uint64_t count_c = 0;
    for(auto const& one_row: v)
    {
        count_c = 0;
        for(auto const& el: one_row)
        {
            array_1d[count_c*nr + count_r] = el; // inversed row <-> column ordering
            ++count_c;
        }
        ++count_r;
    }
}


uint32_t YMIX::round_up_power_2(YCU x)
{
    uint32_t v = x;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

uint64_t YMIX::round_up_power_2(YCUL x)
{
    uint64_t v = x;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}


