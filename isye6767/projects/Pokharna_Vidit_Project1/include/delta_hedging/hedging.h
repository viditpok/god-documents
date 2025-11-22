#pragma once

#include <vector>

namespace delta_hedging
{

    struct HedgingResult
    {
        std::vector<std::vector<double>> hedging_errors;
        std::vector<std::vector<double>> cash_positions;
        std::vector<std::vector<double>> portfolio_values;
        std::vector<std::vector<double>> pnl_option_only;
        std::vector<std::vector<double>> pnl_with_hedge;
    };

    HedgingResult simulate_delta_hedge(const std::vector<std::vector<double>> &stock_paths,
                                       const std::vector<std::vector<double>> &option_prices,
                                       const std::vector<std::vector<double>> &deltas, const std::vector<double> &rate_series,
                                       const std::vector<double> &time_increments);

}
