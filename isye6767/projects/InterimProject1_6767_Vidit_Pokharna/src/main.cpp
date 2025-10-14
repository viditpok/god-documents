#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "delta_hedging/data_io.h"
#include "delta_hedging/hedging.h"
#include "delta_hedging/simulation.h"
#include "delta_hedging/stats.h"

namespace fs = std::filesystem;
using namespace delta_hedging;

namespace
{

    void ensure_directory(const std::string &path)
    {
        fs::create_directories(path);
    }

    void write_matrix_csv(const std::string &filepath, const std::vector<double> &time_grid,
                          const std::vector<std::vector<double>> &matrix, const std::string &first_header)
    {
        std::ofstream out(filepath);
        if (!out)
        {
            throw std::runtime_error("Unable to open csv: " + filepath);
        }
        out << first_header;
        for (std::size_t idx = 0; idx < time_grid.size(); ++idx)
        {
            out << ",t" << idx;
        }
        out << "\n";
        for (std::size_t row = 0; row < matrix.size(); ++row)
        {
            out << first_header << "_" << row;
            for (double value : matrix[row])
            {
                out << "," << value;
            }
            out << "\n";
        }
    }

    void write_summary_csv(const std::string &filepath, const std::vector<double> &values)
    {
        std::ofstream out(filepath);
        if (!out)
        {
            throw std::runtime_error("Unable to open summary csv.");
        }
        out << "metric,value\n";
        out << "mean," << mean(values) << "\n";
        out << "std," << standard_deviation(values) << "\n";
        out << "q05," << quantile(values, 0.05) << "\n";
        out << "median," << quantile(values, 0.50) << "\n";
        out << "q95," << quantile(values, 0.95) << "\n";
    }

    void run_task1(const Task1Config &config, const std::string &output_dir)
    {
        ensure_directory(output_dir);
        SimulationSeries sim =
            simulate_task1_paths(config.s0, config.mu, config.sigma, config.maturity, config.num_steps, config.num_paths,
                                 config.strike, config.rate, config.seed);

        std::vector<double> time_increments(config.num_steps, config.maturity / config.num_steps);
        std::vector<double> rate_series(config.num_steps, config.rate);
        HedgingResult hedge =
            simulate_delta_hedge(sim.stock_paths, sim.option_prices, sim.deltas, rate_series, time_increments);

        write_matrix_csv(output_dir + "/stock_paths.csv", sim.time_grid, sim.stock_paths, "path");
        write_matrix_csv(output_dir + "/option_prices.csv", sim.time_grid, sim.option_prices, "path");
        write_matrix_csv(output_dir + "/deltas.csv", sim.time_grid, sim.deltas, "path");
        write_matrix_csv(output_dir + "/hedging_errors.csv", sim.time_grid, hedge.hedging_errors, "path");
        {
            std::ofstream grid_out(output_dir + "/time_grid.csv");
            if (!grid_out)
            {
                throw std::runtime_error("Unable to write time grid csv.");
            }
            grid_out << "index,time\n";
            for (std::size_t i = 0; i < sim.time_grid.size(); ++i)
            {
                grid_out << i << "," << sim.time_grid[i] << "\n";
            }
        }

        std::vector<double> final_errors;
        final_errors.reserve(hedge.hedging_errors.size());
        for (const auto &row : hedge.hedging_errors)
        {
            if (!row.empty())
            {
                final_errors.push_back(row.back());
            }
        }
        write_summary_csv(output_dir + "/hedging_summary.csv", final_errors);

        std::cout << "Task 1 completed. CSV outputs stored in " << output_dir << std::endl;
    }

    std::string record_date_string(const MarketRecord &record)
    {
        std::ostringstream oss;
        oss << record.year << "-";
        if (record.month < 10)
            oss << "0";
        oss << record.month << "-";
        if (record.day < 10)
            oss << "0";
        oss << record.day;
        return oss.str();
    }

    void write_task2_csv(const std::string &filepath, const std::vector<MarketRecord> &records,
                         const HedgingResult &hedging)
    {
        std::ofstream out(filepath);
        if (!out)
        {
            throw std::runtime_error("Unable to write task2 csv.");
        }
        out << "date,stock,option,implied_vol,delta,hedging_error,pnl_call,pnl_with_hedge,tau,rate\n";
        for (std::size_t i = 0; i < records.size(); ++i)
        {
            const auto &rec = records[i];
            double hedging_error = (i < hedging.hedging_errors[0].size()) ? hedging.hedging_errors[0][i] : 0.0;
            double pnl_call = (i < hedging.pnl_option_only[0].size()) ? hedging.pnl_option_only[0][i] : 0.0;
            double pnl_hedge = (i < hedging.pnl_with_hedge[0].size()) ? hedging.pnl_with_hedge[0][i] : 0.0;
            out << record_date_string(rec) << "," << rec.stock_price << "," << rec.option_price << "," << rec.implied_vol
                << "," << rec.delta << "," << hedging_error << "," << pnl_call << "," << pnl_hedge << "," << rec.tau << ","
                << rec.interest_rate << "\n";
        }
    }

    void run_task2(const Task2Config &config, const std::string &output_dir)
    {
        ensure_directory(output_dir);
        std::vector<MarketRecord> records = load_market_window(config);
        if (records.size() < 2)
        {
            throw std::runtime_error("Insufficient market records for hedging window.");
        }
        std::vector<std::vector<double>> stock_paths(1, std::vector<double>());
        std::vector<std::vector<double>> option_paths(1, std::vector<double>());
        std::vector<std::vector<double>> delta_paths(1, std::vector<double>());
        for (const auto &rec : records)
        {
            stock_paths[0].push_back(rec.stock_price);
            option_paths[0].push_back(rec.option_price);
            delta_paths[0].push_back(rec.delta);
        }
        std::vector<double> rate_series;
        std::vector<double> time_increments;
        rate_series.reserve(records.size() - 1);
        time_increments.reserve(records.size() - 1);
        for (std::size_t i = 1; i < records.size(); ++i)
        {
            const auto &prev = records[i - 1];
            const auto &curr = records[i];
            int bus_days = business_days_between_ymd(prev.year, prev.month, prev.day, curr.year, curr.month, curr.day);
            if (bus_days <= 0)
                bus_days = 1;
            time_increments.push_back(bus_days / 252.0);
            rate_series.push_back(prev.interest_rate);
        }

        HedgingResult hedging =
            simulate_delta_hedge(stock_paths, option_paths, delta_paths, rate_series, time_increments);
        write_task2_csv(output_dir + "/result.csv", records, hedging);

        std::cout << "Task 2 completed. CSV outputs stored in " << output_dir << std::endl;
    }

    void print_usage()
    {
        std::cout << "Usage:\n";
        std::cout << "  delta_hedging task1 [--output DIR] [--paths N]\n";
        std::cout << "  delta_hedging task2 [--output DIR]\n";
        std::cout << "  delta_hedging all\n";
    }

}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        print_usage();
        return 1;
    }
    std::string command = argv[1];
    Task1Config task1_config;
    Task2Config task2_config;
    std::string output_dir = "outputs";
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc)
        {
            output_dir = argv[++i];
        }
        else if (arg == "--paths" && i + 1 < argc)
        {
            task1_config.num_paths = std::stoi(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            task1_config.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        }
        else if (arg == "--start-date" && i + 1 < argc)
        {
            task2_config.start_date = argv[++i];
        }
        else if (arg == "--end-date" && i + 1 < argc)
        {
            task2_config.end_date = argv[++i];
        }
        else if (arg == "--strike" && i + 1 < argc)
        {
            task2_config.strike = std::stod(argv[++i]);
        }
        else if (arg == "--data-dir" && i + 1 < argc)
        {
            task2_config.data_dir = argv[++i];
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
        }
    }
    try
    {
        if (command == "task1")
        {
            run_task1(task1_config, output_dir);
        }
        else if (command == "task2")
        {
            run_task2(task2_config, output_dir);
        }
        else if (command == "all")
        {
            run_task1(task1_config, output_dir);
            run_task2(task2_config, output_dir);
        }
        else
        {
            print_usage();
            return 1;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
