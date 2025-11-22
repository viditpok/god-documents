#include "delta_hedging/stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace delta_hedging
{

    double mean(const std::vector<double> &values)
    {
        if (values.empty())
            return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    }

    double variance(const std::vector<double> &values)
    {
        if (values.size() < 2)
            return 0.0;
        double m = mean(values);
        double accum = 0.0;
        for (double v : values)
        {
            double diff = v - m;
            accum += diff * diff;
        }
        return accum / static_cast<double>(values.size() - 1);
    }

    double standard_deviation(const std::vector<double> &values)
    {
        return std::sqrt(variance(values));
    }

    double quantile(std::vector<double> values, double q)
    {
        if (values.empty())
            return 0.0;
        if (q <= 0.0)
            return *std::min_element(values.begin(), values.end());
        if (q >= 1.0)
            return *std::max_element(values.begin(), values.end());
        std::sort(values.begin(), values.end());
        double idx = q * (values.size() - 1);
        std::size_t lower = static_cast<std::size_t>(idx);
        std::size_t upper = std::min(lower + 1, values.size() - 1);
        double weight = idx - lower;
        return values[lower] * (1.0 - weight) + values[upper] * weight;
    }

}
