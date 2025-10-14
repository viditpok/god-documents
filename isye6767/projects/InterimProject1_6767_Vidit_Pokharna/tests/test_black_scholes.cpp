#include <cassert>
#include <cmath>
#include <iostream>

#include "delta_hedging/black_scholes.h"

using namespace delta_hedging;

int main()
{
    double price = black_scholes_call(100.0, 105.0, 0.01, 0.2, 0.5);
    assert(std::abs(price - 3.7988) < 1e-3);
    double delta = black_scholes_delta(110.0, 100.0, 0.02, 0.25, 0.75);
    assert(delta > 0.5 && delta < 1.0);
    double market = black_scholes_call(110.0, 100.0, 0.02, 0.35, 0.75);
    double iv = implied_volatility(market, 110.0, 100.0, 0.02, 0.75);
    assert(std::abs(iv - 0.35) < 1e-3);
    std::cout << "All Black-Scholes tests passed.\n";
    return 0;
}
