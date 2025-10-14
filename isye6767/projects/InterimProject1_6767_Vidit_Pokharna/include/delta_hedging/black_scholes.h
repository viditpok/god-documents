#pragma once

namespace delta_hedging
{

    double norm_cdf(double x);

    double black_scholes_call(double spot, double strike, double rate, double volatility, double time_to_maturity);

    double black_scholes_delta(double spot, double strike, double rate, double volatility, double time_to_maturity);

    double implied_volatility(double market_price, double spot, double strike, double rate, double time_to_maturity,
                              double tolerance = 1e-6, int max_iterations = 200);

}
