#pragma once

#include <vector>

namespace delta_hedging
{

    struct SimulationSeries
    {
        std::vector<std::vector<double>> stock_paths;
        std::vector<std::vector<double>> option_prices;
        std::vector<std::vector<double>> deltas;
        std::vector<double> time_grid;
    };

    SimulationSeries simulate_task1_paths(double s0, double mu, double sigma, double maturity, int num_steps,
                                          int num_paths, double strike, double rate, unsigned int seed);

}
