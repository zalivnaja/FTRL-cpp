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

#include "MurmurHash3.cc"

using namespace std;


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

std::string get_hash_feature(const std::string& _value, uint32_t _hash_size)
{
    static std::hash<std::string> hash_fn;
    uint32_t hash_value = std::abs(hash_fn(_value));

    size_t hash = hash_value % _hash_size;
    return to_string(hash);
}


struct Row
{
public:

    void fillFromStr(const std::vector<std::string>& _columns, const std::string& _row, int _categ_features_count, int _hash_size)
    {
        std::istringstream ss(_row);
        std::string token;

        int comp_count = _columns.size();

        int current_comp = 0;
        while(std::getline(ss, token, ',')) 
        {
            if (current_comp + 1 < comp_count)
            {
                const auto& value = token;

                if (current_comp == 0)
                {
                    ++current_comp;
                    continue;
                }

                if (current_comp < _columns.size() - 1 - _categ_features_count)
                {
                    components_[_columns[current_comp]] = ::atof(value.c_str());
                }
                else
                {
                    components_[_columns[current_comp] + "_" + get_hash_feature(value, _hash_size)] = 1;
                }

                ++current_comp;
            } else {
                target_ = ::atof(token.c_str());
            }
        }
    }

    Row()
    {
    }

    std::unordered_map<string, float> components_;
    float target_;
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
    DataProvider(const std::string& _filename, int _categ_features_count, int _bits)
        : filename_(_filename)
        , data_file_(filename_)
        , categ_features_count_(_categ_features_count)
        , bits_(_bits)
    {
        if (data_file_.is_open())
        {
            std::string columns_row;
            if (!getline(data_file_, columns_row))
            {
                last_error_ = "cann't read head";
                data_file_.close();
            }

            columns_ = split(columns_row, ',');
            numerical_columns_count_ = columns_.size() - 2 - _categ_features_count; // id and target
            hashed_features_count_ = get_hashed_features_count();
            // std::cout << "columns_count_ " << columns_count_ << std::endl;
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
            if (getline(data_file_, line_))
            {
                _row->fillFromStr(columns_, line_, categ_features_count_, hashed_features_count_);
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
        auto result = (2 << bits_) - numerical_columns_count_;
        if (result < 0)
        {
            throw std::runtime_error("Too small amount of bits.");
        }

        return result; 
    }

    int bits_;
    std::vector<std::string> columns_;
    int numerical_columns_count_;
    int hashed_features_count_;
};


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

struct NZ
{
    float n;
    float z;
};


class FTRL
{
public:
    FTRL(int _bits, float _alpha, float _beta, float _L1, float _L2, int _numerical_columns_count
        , std::vector<std::string> _columns, int _categ_features_count)
        
        : bits_(_bits)
        , alpha_(_alpha)
        , beta_(_beta)
        , L1_(_L1)
        , L2_(_L2)
        , numerical_columns_count_(_numerical_columns_count)
        , columns_(_columns)
        , categ_features_count_(_categ_features_count)
    {
    }

    float predict(const Row& _x)
    {
        w_.clear();

        float wx = 0;

        for (auto& x_i : _x.components_)
        {
            auto& i = x_i.first;

            auto nz_i = nz_[i];
            auto z_i = nz_i.z;
            auto n_i = nz_i.n;

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
            w_[i] = new_w_i;
            wx += new_w_i * x_i.second;
        }

        return sigmoid(wx);
    }

    void update(const Row& x, float p, float y)
    {
        float g = p - y;

        for (auto& comp : x.components_)
        {
            auto& i = comp.first;
            auto g_i = g * comp.second;
            
            auto nz_i = nz_[i];
            auto z_i = nz_i.z;
            auto n_i = nz_i.n;

            auto sigma_i = (sqrt(n_i + g_i * g_i) - sqrt(n_i)) / alpha_;
            z_i += g_i - sigma_i * w_[i];
            n_i += g_i * g_i;
            NZ new_nz;
            new_nz.z = z_i;
            new_nz.n = n_i;
            nz_[i] = new_nz;
        }
    }

private:
    int bits_;
    float alpha_;
    float beta_;
    float L1_;
    float L2_;
    int numerical_columns_count_;
    int categ_features_count_;
    std::vector<std::string> columns_;

    unordered_map<string, NZ> nz_;
    unordered_map<string, float> w_;
};

int main()
{
    int bits = 4;

    // TODO : unhard-code it
    // const string& filename = "categ_testdata0.csv";
    // int categ_features_count = 0;

    const string& filename = "categ_testdata.csv";
    int categ_features_count = 10;

    DataProvider dataProvider(filename, categ_features_count, bits);
    FTRL ftrl(bits, .005, 1, 0, 1, dataProvider.numerical_columns_count_, dataProvider.columns_, categ_features_count);

    Row row;
    int i = 0;
    int passes = 10;

    int t_predict = 0;   // get time now
    int t_logloss = 0;   // get time now
    int t_update = 0;   // get time now
    time_t t0 = time(0);   // get time now

    for (int pass_i = 0; pass_i < passes; ++pass_i)
    {
        float loss = 0;

        DataProvider dataProvider(filename, categ_features_count, bits);
        while (dataProvider.TryGetNextRow(&row))
        {
            time_t t1 = time(0);   // get time now
            float p = ftrl.predict(row);
            t_predict += time(0) - t1;

            t1 = time(0);
            loss += logloss(p, row.target_);
            t_logloss += time(0) - t1;

            t1 = time(0);
            ftrl.update(row, p, row.target_);
            t_update += time(0) - t1;
        }

        std::cout << pass_i << ", " << loss << std::endl;
    }
    std::cout << "t_predict" << t_predict << std::endl;
    std::cout << "t_logloss" << t_logloss << std::endl;
    std::cout << "t_update" << t_update << std::endl;
    std::cout << time(0) - t0 << std::endl;

    return 0;
}
