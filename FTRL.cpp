#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;


struct Row
{

public:
    void fillFromStr(const std::string& _columns, const std::string& _row)
    {
        std::istringstream ss(_row);
        std::string token;

        int comp_count = std::count(_row.begin(), _row.end(), ',');
        components_.resize(comp_count);

        int current_comp = 0;
        while(std::getline(ss, token, ',')) 
        {
            if (current_comp + 1 < comp_count)
                components_[current_comp++] = ::atof(token.c_str());
            else
                target_ = ::atof(token.c_str());
        }
    }

    Row()
    {
    }

    std::vector<float> components_;
    float target_;
};

class DataProvider
{
private:
    std::string filename_;
    ifstream data_file_;
    std::string line_;
    std::string columns_;

    std::string last_error_;

public:
    DataProvider(const std::string& _filename)
        : filename_(_filename)
        , data_file_(filename_)
    {
        if (data_file_.is_open())
        {
            if (!getline(data_file_, columns_))
            {
                last_error_ = "cann't read head";
                data_file_.close();
            }

            columns_count_ = std::count(columns_.begin(), columns_.end(), ',') - 1; // id and target
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
                _row->fillFromStr(columns_, line_);
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

    int columns_count_;
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


class FTRL
{
public:
    FTRL(float _alpha, float _beta, float _L1, float _L2, int _columns_count)
        : alpha_(_alpha)
        , beta_(_beta)
        , L1_(_L1)
        , L2_(_L2)
        , columns_count_(_columns_count)
        , n_(columns_count_, 0.0)
        , z_(columns_count_, 0.0)
    {}

    float predict(const Row& _x)
    {
        w_.clear();

        float wx = 0;
        int i = 0;
        for (auto x_i : _x.components_)
        {
            if (abs(z_[i]) <= L1_)
            {
                w_[i] = 0;
            }
            else
            {
                float sign = z_[i] < 0 ? -1 : 1;
                w_[i] = (sign * L1_ - z_[i]) / ((beta_ + sqrt(n_[i])) / alpha_ + L2_);
            }
            wx += w_[i] * x_i;
            ++i;
        }

        return sigmoid(wx);
    }

    void update(const Row& x, float p, float y)
    {
        float g = p - y;

        int i = 0;
        for (auto comp : x.components_)
        {
            auto g_i = g * comp;
            auto sigma_i = (sqrt(n_[i] + g_i * g_i) - sqrt(n_[i])) / alpha_;
            z_[i] += g_i - sigma_i * w_[i];
            n_[i] += g_i * g_i;
            ++i;
        }
    }

private:
    float alpha_;
    float beta_;
    float L1_;
    float L2_;
    int columns_count_;

    vector<float> n_;
    vector<float> z_;
    unordered_map<int, float> w_;
};

int main()
{
    DataProvider dataProvider("testdata.csv");
    FTRL ftrl(.005, 1, 0, 1, dataProvider.columns_count_);

    Row row;
    int i = 0;
    int passes = 1000;

    for (int pass_i = 0; pass_i < passes; ++pass_i)
    {
        float loss = 0;

        DataProvider dataProvider("testdata.csv");
        while (dataProvider.TryGetNextRow(&row))
        {
            float p = ftrl.predict(row);
            loss += logloss(p, row.target_);
            ftrl.update(row, p, row.target_);
        }

        std::cout << pass_i << ", " << loss << std::endl;
    }

    return 0;
}
