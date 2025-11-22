#pragma once

#include <string>
#include <vector>

namespace delta_hedging
{

    struct MarketRecord
    {
        int year;
        int month;
        int day;
        double stock_price;
        double option_price;
        double interest_rate;
        double tau;
        double implied_vol;
        double delta;
    };

    struct Task2Config
    {
        std::string data_dir{"data"};
        double strike{500.0};
        std::string expiry{"2011-09-17"};
        std::string cp_flag{"C"};
        std::string start_date{"2011-07-05"};
        std::string end_date{"2011-07-29"};
    };

    struct Task1Config
    {
        double s0{100.0};
        double mu{0.05};
        double sigma{0.24};
        double rate{0.025};
        double maturity{0.4};
        int num_steps{100};
        int num_paths{1000};
        double strike{105.0};
        unsigned int seed{42};
    };

    std::vector<MarketRecord> load_market_window(const Task2Config &config);
    int business_days_between_ymd(int y1, int m1, int d1, int y2, int m2, int d2);

}
