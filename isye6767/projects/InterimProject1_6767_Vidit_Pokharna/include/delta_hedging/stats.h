#pragma once

#include <vector>

namespace delta_hedging
{

    double mean(const std::vector<double> &values);
    double variance(const std::vector<double> &values);
    double standard_deviation(const std::vector<double> &values);
    double quantile(std::vector<double> values, double q);

}
