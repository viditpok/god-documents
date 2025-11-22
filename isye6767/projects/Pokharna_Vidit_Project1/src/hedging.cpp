#include "delta_hedging/hedging.h"

#include <cmath>
#include <stdexcept>

namespace delta_hedging
{

    namespace
    {
        void validate_dimensions(const std::vector<std::vector<double>> &stock_paths,
                                 const std::vector<std::vector<double>> &option_prices,
                                 const std::vector<std::vector<double>> &deltas)
        {
            if (stock_paths.size() != option_prices.size() || stock_paths.size() != deltas.size())
            {
                throw std::invalid_argument("Input matrices must share identical shapes.");
            }
            for (std::size_t i = 0; i < stock_paths.size(); ++i)
            {
                if (stock_paths[i].size() != option_prices[i].size() || stock_paths[i].size() != deltas[i].size())
                {
                    throw std::invalid_argument("Row length mismatch in hedging inputs.");
                }
            }
        }
    }

    HedgingResult simulate_delta_hedge(const std::vector<std::vector<double>> &stock_paths,
                                       const std::vector<std::vector<double>> &option_prices,
                                       const std::vector<std::vector<double>> &deltas, const std::vector<double> &rate_series,
                                       const std::vector<double> &time_increments)
    {
        validate_dimensions(stock_paths, option_prices, deltas);
        if (stock_paths.empty())
        {
            throw std::invalid_argument("At least one path must be supplied.");
        }
        const std::size_t num_steps = stock_paths.front().size() - 1;
        if (rate_series.size() != num_steps || time_increments.size() != num_steps)
        {
            throw std::invalid_argument("Rate series and time increments must provide values for each step.");
        }

        HedgingResult result;
        result.hedging_errors.assign(stock_paths.size(), std::vector<double>(num_steps + 1, 0.0));
        result.cash_positions.assign(stock_paths.size(), std::vector<double>(num_steps + 1, 0.0));
        result.portfolio_values.assign(stock_paths.size(), std::vector<double>(num_steps + 1, 0.0));
        result.pnl_option_only.assign(stock_paths.size(), std::vector<double>(num_steps + 1, 0.0));
        result.pnl_with_hedge.assign(stock_paths.size(), std::vector<double>(num_steps + 1, 0.0));

        for (std::size_t path = 0; path < stock_paths.size(); ++path)
        {
            const auto &stock_row = stock_paths[path];
            const auto &option_row = option_prices[path];
            const auto &delta_row = deltas[path];
            auto &hedge_row = result.hedging_errors[path];
            auto &cash_row = result.cash_positions[path];
            auto &port_row = result.portfolio_values[path];
            auto &pnl_option_row = result.pnl_option_only[path];
            auto &pnl_hedge_row = result.pnl_with_hedge[path];

            cash_row[0] = option_row[0] - delta_row[0] * stock_row[0];
            port_row[0] = delta_row[0] * stock_row[0] + cash_row[0];
            hedge_row[0] = port_row[0] - option_row[0];

            for (std::size_t step = 1; step <= num_steps; ++step)
            {
                const double prev_delta = delta_row[step - 1];
                const double prev_cash = cash_row[step - 1];
                const double rate = rate_series[step - 1];
                const double dt = time_increments[step - 1];
                const double grown_cash = prev_cash * std::exp(rate * dt);
                const double stock_value = stock_row[step];
                const double option_value = option_row[step];
                const double hedged_value = prev_delta * stock_value + grown_cash;

                hedge_row[step] = hedged_value - option_value;

                const double new_delta = delta_row[step];
                cash_row[step] = hedged_value - new_delta * stock_value;
                port_row[step] = new_delta * stock_value + cash_row[step];
                pnl_option_row[step] = option_row[0] - option_value;
                pnl_hedge_row[step] = hedge_row[step];
            }
        }

        return result;
    }

}
