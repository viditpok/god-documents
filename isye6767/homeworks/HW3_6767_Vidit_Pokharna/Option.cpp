#include "Option.h"
#include <iostream>

Option::Option() { init(); }

Option::Option(double strike, double spot, double rate, double time, double vol)
    : K(strike), S(spot), r(rate), T(time), sigma(vol) {}

Option::~Option() {}

void Option::init() {
  K = 100.0;
  S = 100.0;
  r = 0.05;
  T = 1.0;
  sigma = 0.2;
}

double Option::getStrikePrice() const { return K; }

double Option::getCurrentPrice() const { return S; }

double Option::getRiskFreeRate() const { return r; }

double Option::getTimeToMaturity() const { return T; }

double Option::getVolatility() const { return sigma; }
