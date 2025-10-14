#include "delta_hedging/simulation.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "delta_hedging/black_scholes.h"

namespace delta_hedging
{

    SimulationSeries simulate_task1_paths(double s0, double mu, double sigma, double maturity, int num_steps,
                                          int num_paths, double strike, double rate, unsigned int seed)
    {
        if (num_steps <= 0 || num_paths <= 0)
        {
            throw std::invalid_argument("num_steps and num_paths must be positive.");
        }
        const double dt = maturity / static_cast<double>(num_steps);
        const double drift = (mu - 0.5 * sigma * sigma) * dt;
        const double diffusion = sigma * std::sqrt(dt);

        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        SimulationSeries series;
        series.stock_paths.assign(num_paths, std::vector<double>(num_steps + 1));
        series.option_prices.assign(num_paths, std::vector<double>(num_steps + 1));
        series.deltas.assign(num_paths, std::vector<double>(num_steps + 1));
        series.time_grid.resize(num_steps + 1);
        for (int step = 0; step <= num_steps; ++step)
        {
            series.time_grid[static_cast<std::size_t>(step)] = dt * step;
        }
        for (int path = 0; path < num_paths; ++path)
        {
            double s = s0;
            series.stock_paths[path][0] = s;
            series.option_prices[path][0] = black_scholes_call(s, strike, rate, sigma, maturity);
            series.deltas[path][0] = black_scholes_delta(s, strike, rate, sigma, maturity);
            for (int step = 1; step <= num_steps; ++step)
            {
                double z = dist(rng);
                s = s * std::exp(drift + diffusion * z);
                series.stock_paths[path][step] = s;
                double remaining = maturity - series.time_grid[static_cast<std::size_t>(step)];
                remaining = std::max(remaining, 0.0);
                series.option_prices[path][step] = black_scholes_call(s, strike, rate, sigma, remaining);
                series.deltas[path][step] = black_scholes_delta(s, strike, rate, sigma, remaining);
            }
        }
        return series;
    }

}
