#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <exception>
#include <ctime>

#include <streambuf>
#include <dirent.h> 
#include <stdio.h> 

using namespace std;


namespace MLUtils
{
    float sigmoid(float _x, float _eps = 35.0)
    {
        _x = max(min(_x, _eps), -_eps);

        if (_x >= 0)
        {
            float z = exp(-_x);
            return 1 / (1 + z);
        }
        else
        {
            float z = exp(_x);
            return z / (1 + z);
        }
    }

    float logloss(float _p, float _y, float _eps = 1e-5)
    {
        _p = max(min(_p, 1 - _eps), _eps);
        return _y == 1 ? -log(_p) : -log(1 - _p);
    }

    float cut_result(float _result, float _eps = 1e-3)
    {
        return max(min(_result, 1 - _eps), _eps);
    }
};

namespace Utils
{
    template<typename Out>
    void split(const std::string &s, char delim, Out result) {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

    std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    // http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
    std::istream& safeGetline(std::istream& is, std::string& t)
    {
        t.clear();

        // The characters in the stream are read one-by-one using a std::streambuf.
        // That is faster than reading them one-by-one using the std::istream.
        // Code that uses streambuf this way must be guarded by a sentry object.
        // The sentry object performs various tasks,
        // such as thread synchronization and updating the stream state.

        std::istream::sentry se(is, true);
        std::streambuf* sb = is.rdbuf();

        for(;;) {
            int c = sb->sbumpc();
            switch (c) {
            case '\n':
                return is;
            case '\r':
                if(sb->sgetc() == '\n')
                    sb->sbumpc();
                return is;
            case EOF:
                // Also handle the case when the last line has no line ending
                if(t.empty())
                    is.setstate(std::ios::eofbit);
                return is;
            default:
                t += (char)c;
            }
        }
    }

    class InputParser{
        public:
            InputParser (int &argc, char **argv){
                for (int i=1; i < argc; ++i)
                    this->tokens.push_back(std::string(argv[i]));
            }
            const std::string& getCmdOption(const std::string &option) const{
                std::vector<std::string>::const_iterator itr;
                itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
                if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                    return *itr;
                }
                static const std::string empty_string("");
                return empty_string;
            }
            bool cmdOptionExists(const std::string &option) const{
                return std::find(this->tokens.begin(), this->tokens.end(), option)
                       != this->tokens.end();
            }
        private:
            std::vector <std::string> tokens;
    };
}

size_t get_hash_feature(const std::string& _value, uint32_t _hash_size)
{
    static std::hash<std::string> hash_fn;

    uint32_t hash_value = hash_fn(_value);
    return hash_value % _hash_size;
}

struct RowInfo
{
public:
    RowInfo()
    {}

    RowInfo(int _target_column_ind, int _id_column_ind, int _columns_count, int _categ_features_count)
        : target_column_ind_(_target_column_ind)
        , id_column_ind_(_id_column_ind)
        , columns_count_(_columns_count)
    {
        int add_id_count = 0;
        if (id_column_ind_ >=0) 
            add_id_count += 1;

        int add_target_count = 0;
        if (target_column_ind_ >=0) 
            add_target_count += 1;

        numerical_features_count_ = columns_count_ - _categ_features_count - add_target_count - add_id_count;
    }

    bool is_numerical_feature(int _column_ind) const
    {
        return _column_ind != target_column_ind_
            && _column_ind != id_column_ind_
            && _column_ind < numerical_features_count_;
    }

    bool is_categorical_feature(int _column_ind) const
    {
        return _column_ind >= numerical_features_count_;
    }

    int target_column_ind_;
    int id_column_ind_;
    int numerical_features_count_;
    
private:
    int columns_count_;
};

struct Row
{
public:

    void fillFromStr(const std::vector<std::string>& _columns, const std::string& _row, RowInfo _row_info, int _hash_size)
    {
        components_keys_.clear();
        components_values_.clear();
        
        std::istringstream ss(_row);
        std::string token;

        int current_comp = 0;
        while(std::getline(ss, token, ',')) 
        {
            const auto& value = token;

            if (current_comp == _row_info.id_column_ind_)
            {
                id_ = token;
            }
            else if (current_comp == _row_info.target_column_ind_)
            {
                target_ = ::atof(token.c_str());
            }
            else if (_row_info.is_numerical_feature(current_comp))
            {
                components_values_.push_back(::atof(value.c_str()));
                components_keys_.push_back(current_comp);
            }
            else
            {
                auto key = _row_info.numerical_features_count_ + get_hash_feature(_columns[current_comp] + ";" + value, _hash_size);

                components_values_.push_back(1);
                components_keys_.push_back(key);
            }

            ++current_comp;
        }
    }

    Row()
    {
    }

    std::vector<int> components_keys_;
    std::vector<float> components_values_;

    float target_;
    string id_;
};

class DataProvider
{
private:
    std::string filename_;
    ifstream data_file_;
    std::string line_;

    std::string last_error_;
    int categ_features_count_;

public:
    DataProvider(const std::string& _filename, int _categ_features_count, int _bits, int _id_column_ind, int _target_column_ind)
        : filename_(_filename)
        , data_file_(filename_)
        , categ_features_count_(_categ_features_count)
        , bits_(_bits)
    {
        if (data_file_.is_open())
        {
            std::string columns_row;
            if (Utils::safeGetline(data_file_, columns_row).eof())
            {
                last_error_ = "cann't read head";
                data_file_.close();
            }

            columns_ = Utils::split(columns_row, ',');
            row_info_ = RowInfo(_target_column_ind, _id_column_ind, columns_.size(), _categ_features_count);
            hashed_features_count_ = get_hashed_features_count();
        }
        else
        {
            last_error_ = "cann't open file";
        }
    }

    bool TryGetNextRow(Row* _row)
    {
        if (data_file_.is_open())
        {
            if (!Utils::safeGetline(data_file_, line_).eof())
            {
                _row->fillFromStr(columns_, line_, row_info_, hashed_features_count_);
                return true;
            }
            else
            {
                data_file_.close();
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    int get_hashed_features_count() const
    {
        auto result = (2 << bits_) - row_info_.numerical_features_count_;
        if (result < 0)
        {
            throw std::runtime_error("Too small amount of bits.");
        }

        return result; 
    }

    int bits_;
    std::vector<std::string> columns_;
    int hashed_features_count_;

    RowInfo row_info_;
};



class FTRL
{
public:
    FTRL(int _bits, float _alpha, float _beta, float _L1, float _L2)
        : bits_(_bits)
        , alpha_(_alpha)
        , beta_(_beta)
        , L1_(_L1)
        , L2_(_L2)
        , n_(2 << bits_, 0.0)
        , z_(2 << bits_, 0.0)
    {
    }

    FTRL()
    {}

    float predict(const Row& _x, bool train = true)
    {
        if (train)
            w_.clear();

        float wx = 0;

        for (int i = 0; i < _x.components_keys_.size(); ++i)
        {
            auto& key = _x.components_keys_[i];
            auto& value = _x.components_values_[i];

            auto z_i = z_[key];
            auto n_i = n_[key];

            auto new_w_i = 0.0;

            if (abs(z_i) <= L1_)
            {
                new_w_i = 0;
            }
            else
            {
                float sign = z_i < 0 ? -1 : 1;
                new_w_i = (sign * L1_ - z_i) / ((beta_ + sqrt(n_i)) / alpha_ + L2_);
            }
            if (train)
                w_[key] = new_w_i;
            wx += new_w_i * value;
        }

        return MLUtils::sigmoid(wx);
    }

    void update(const Row& _x, float _p, float _y)
    {
        float g = _p - _y;

        for (int i = 0; i < _x.components_keys_.size(); ++i)
        {
            auto& key = _x.components_keys_[i];
            auto& value = _x.components_values_[i];

            auto g_i = g * value;
            auto n_i = n_[key];

            auto sigma_i = (sqrt(n_i + g_i * g_i) - sqrt(n_i)) / alpha_;
            z_[key] += g_i - sigma_i * w_[key];
            n_[key] += g_i * g_i;
        }
    }

    void save_to_file(const string& _filename) const
    {
        ofstream file;
        file.open(_filename);

        file << bits_ << std::endl;
        file << alpha_ << std::endl;
        file << beta_ << std::endl;
        file << L1_ << std::endl;
        file << L2_ << std::endl;

        file << w_.size() << std::endl;
        for (auto& w : w_)
        {
            file << w.first << " " << w.second << std::endl;
        }

        file << n_.size() << std::endl;
        for (auto& n : n_)
        {
            file << n << std::endl;
        }

        file << z_.size() << std::endl;
        for (auto& n : z_)
        {
            file << n << std::endl;
        }

        file.close();
    }

    // TODO : add try-catch
    void load_from_file(const string& _filename)
    {
        ifstream file;
        file.open(_filename);

        file >> bits_;
        file >> alpha_;
        file >> beta_;
        file >> L1_;
        file >> L2_;

        int w_size;
        file >> w_size;
        for (int i = 0; i < w_size; ++i)
        {
            int column;
            float w;

            file >> column >> w;
            w_[column] = w;
        }

        int n_size;
        file >> n_size;
        n_.resize(n_size);
        for (int i = 0; i < n_size; ++i)
        {
            float n;
            file >> n;
            n_[i] = n;
        }

        int z_size;
        file >> z_size;
        z_.resize(z_size);
        for (int i = 0; i < z_size; ++i)
        {
            float n;
            file >> n;
            z_[i] = n;
        }

        file.close();
    }

private:
    int bits_;
    float alpha_;
    float beta_;
    float L1_;
    float L2_;

    vector<float> n_;
    vector<float> z_;
    unordered_map<int, float> w_;
};


struct Params
{
    bool is_testing;
    int bits;
    string model_filename;
    string data_filename;
    string test_predictions_filename;
    int categ_features_count;

    int id_column_ind;
    int target_column_ind;
    int passes;
};


// TODO : it is ugly version
// add try-catch, train-test separation of params, ..
bool try_get_params_from_command_line(Params& params, int argc, char* argv[])
{
    Utils::InputParser input(argc, argv);
    const auto& usage_helper = "Usage is -t <is_testing> -b <bits count> -fm <model_filename>\
         -fd <datafilename> -ftr <test_predictions_filename>\n";

    if (argc < 3)
    {
        std::cout << "argc < 3: " << argc << std::endl;
        std::cout << usage_helper;
        return false;
    }

    const string& is_testing = input.getCmdOption("-t");
    if (is_testing.empty())
    {
        std::cout << usage_helper;
        return false;
    }
    params.is_testing = is_testing == "1";

    params.bits = 10; // default bits count
    const string& bits_argv = input.getCmdOption("-b");
    if (!bits_argv.empty())
    {
        params.bits = std::stoi(bits_argv);
    }

    params.model_filename = "my_ftrl_model";
    const string& model_filename_argv = input.getCmdOption("-fm");
    if (!model_filename_argv.empty())
    {
        params.model_filename = model_filename_argv;
    }

    params.data_filename = params.is_testing ? "test.csv" : "train.csv";
    const string& data_filename_argv = input.getCmdOption("-fd");
    if (!data_filename_argv.empty())
    {
        params.data_filename = data_filename_argv;
    }

    params.test_predictions_filename = "result.csv";
    const string& test_predictions_filename_argv = input.getCmdOption("-ftr");
    if (!test_predictions_filename_argv.empty())
    {
        params.test_predictions_filename = test_predictions_filename_argv;
    }

    params.categ_features_count = 22;
    const string& categ_features_count_argv = input.getCmdOption("-cc");
    if (!categ_features_count_argv.empty())
    {
        params.categ_features_count = std::stoi(categ_features_count_argv);
    }

    if (params.is_testing)
    {
        params.id_column_ind = 0;
        params.target_column_ind = -1;
    }
    else
    {
        params.id_column_ind = 0;
        params.target_column_ind = 1;
    }
    params.passes = 1;

    return true;
}

void train_and_save_model(const Params& _params)
{
    FTRL ftrl(_params.bits, .005, 1, 0, 1);
    Row row;

    for (int pass_i = 0; pass_i < _params.passes; ++pass_i)
    {
        float loss = 0;
        DataProvider dataProvider(_params.data_filename, _params.categ_features_count, _params.bits\
            , _params.id_column_ind, _params.target_column_ind);

        int lines_count = 0;
        while (dataProvider.TryGetNextRow(&row))
        {
            if (lines_count % 100000 == 0)
            {
                std::cout << "lines_count " << lines_count << " " << loss / (lines_count + 1) << std::endl;
            }

            // early stopping for test
            // if (lines_count > 500000)
            //    break;

            ++lines_count;

            float p = ftrl.predict(row);
            loss += MLUtils::logloss(p, row.target_);
            ftrl.update(row, p, row.target_);
        }

        std::cout << pass_i << ", " << loss << std::endl;
    }
    ftrl.save_to_file(_params.model_filename);
}

void load_model_and_calc_test(const Params& _params)
{
    FTRL ftrl;
    ftrl.load_from_file(_params.model_filename);

    ofstream result_file;
    result_file.open(_params.test_predictions_filename);

    DataProvider dataProvider(_params.data_filename, _params.categ_features_count, _params.bits\
        , _params.id_column_ind, _params.target_column_ind);

    Row row;
    int lines_count = 0;
    while (dataProvider.TryGetNextRow(&row))
    {
        if (lines_count % 100000 == 0)
        {
            std::cout << "lines_count " << lines_count << std::endl;
        }
        ++lines_count;

        float p = ftrl.predict(row, false);
        result_file << row.id_ << "," << MLUtils::cut_result(p) << std::endl;
    }

    result_file.close();
}

int main(int argc, char* argv[])
{
    Params params;
    if (!try_get_params_from_command_line(params, argc, argv))
    {
        return 0;
    }
    
    if (!params.is_testing)
    {
        train_and_save_model(params);
    }
    else
    {
        load_model_and_calc_test(params);
    }

    return 0;
}
