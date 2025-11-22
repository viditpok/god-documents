#include "delta_hedging/black_scholes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace delta_hedging
{

    namespace
    {
        constexpr double SQRT_TWO = 1.41421356237309504880;

        void validate_strike(double strike)
        {
            if (strike <= 0.0)
            {
                throw std::invalid_argument("Strike must be positive.");
            }
        }
    }

    double norm_cdf(double x)
    {
        return 0.5 * (1.0 + std::erf(x / SQRT_TWO));
    }

    double black_scholes_call(double spot, double strike, double rate, double volatility, double time_to_maturity)
    {
        validate_strike(strike);
        double spot_safe = std::max(spot, 1e-12);
        if (time_to_maturity <= 0.0 || volatility <= 0.0)
        {
            return std::max(spot_safe - strike, 0.0);
        }

        const double sqrt_tau = std::sqrt(time_to_maturity);
        const double sigma_sq = volatility * volatility;
        const double d1 = (std::log(spot_safe / strike) + (rate + 0.5 * sigma_sq) * time_to_maturity) / (volatility * sqrt_tau);
        const double d2 = d1 - volatility * sqrt_tau;
        return spot_safe * norm_cdf(d1) - strike * std::exp(-rate * time_to_maturity) * norm_cdf(d2);
    }

    double black_scholes_delta(double spot, double strike, double rate, double volatility, double time_to_maturity)
    {
        validate_strike(strike);
        if (time_to_maturity <= 0.0 || volatility <= 0.0)
        {
            return spot > strike ? 1.0 : 0.0;
        }
        const double sqrt_tau = std::sqrt(time_to_maturity);
        const double sigma_sq = volatility * volatility;
        const double d1 = (std::log(std::max(spot, 1e-12) / strike) + (rate + 0.5 * sigma_sq) * time_to_maturity) /
                          (volatility * sqrt_tau);
        return norm_cdf(d1);
    }

    double implied_volatility(double market_price, double spot, double strike, double rate, double time_to_maturity,
                              double tolerance, int max_iterations)
    {
        validate_strike(strike);
        if (time_to_maturity <= 0.0)
        {
            return 0.0;
        }
        double spot_safe = std::max(spot, 1e-12);
        double intrinsic = std::max(spot_safe - strike * std::exp(-rate * time_to_maturity), 0.0);
        double upper_bound = spot_safe;
        if (market_price < intrinsic - tolerance || market_price > upper_bound + tolerance)
        {
            throw std::invalid_argument("Market price violates arbitrage bounds.");
        }
        double low = 1e-6;
        double high = 5.0;
        double last_mid = low;
        for (int i = 0; i < max_iterations; ++i)
        {
            double mid = 0.5 * (low + high);
            double price = black_scholes_call(spot_safe, strike, rate, mid, time_to_maturity);
            double diff = price - market_price;
            if (std::abs(diff) < tolerance)
            {
                return mid;
            }
            if (diff > 0)
            {
                high = mid;
            }
            else
            {
                low = mid;
            }
            last_mid = mid;
        }
        return last_mid;
    }

}
