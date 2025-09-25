#include "Option_Price.h"
#include <cmath>
#include <iomanip>
#include <iostream>

Option_Price::Option_Price(double strike, double spot, double rate, double time,
                           double vol, char optionFlag)
    : Option(strike, spot, rate, time, vol), flag(optionFlag) {}

Option_Price::~Option_Price() {}

char Option_Price::getFlag() const { return flag; }

double Option_Price::normalCDF(double x) {
  return 0.5 * (1 + erf(x / sqrt(2.0)));
}

double Option_Price::normalPDF(double x) {
  return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
}

double Option_Price::binomialCoeff(int n, int k) {
  if (k > n)
    return 0;
  if (k == 0 || k == n)
    return 1;

  double result = 1;
  for (int i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

double Option_Price::binomialPrice(int n, double u, double d, double p,
                                   double q) {
  double S = getCurrentPrice();
  double K = getStrikePrice();
  double r = getRiskFreeRate();
  double T = getTimeToMaturity();

  double price = 0.0;
  for (int j = 0; j <= n; j++) {
    double stockPrice = S * pow(u, j) * pow(d, n - j);
    double payoff;

    if (flag == 'c' || flag == 'C') {
      payoff = std::max(stockPrice - K, 0.0);
    } else {
      payoff = std::max(K - stockPrice, 0.0);
    }

    price += binomialCoeff(n, j) * pow(p, j) * pow(q, n - j) * payoff;
  }
  return price * exp(-r * T);
}

double Option_Price::BSM_Pricer() {
  double S = getCurrentPrice();
  double K = getStrikePrice();
  double r = getRiskFreeRate();
  double T = getTimeToMaturity();
  double sigma = getVolatility();

  double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
  double d2 = d1 - sigma * sqrt(T);

  double price;
  if (flag == 'c' || flag == 'C') {
    price = S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
  } else {
    price = K * exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
  }

  return price;
}

double Option_Price::Binomial_Pricer() {
  double T = getTimeToMaturity();
  double sigma = getVolatility();
  double r = getRiskFreeRate();

  int n = 100;
  double dt = T / n;
  double u = exp(sigma * sqrt(dt));
  double d = 1.0 / u;
  double p = (exp(r * dt) - d) / (u - d);
  double q = 1.0 - p;

  return binomialPrice(n, u, d, p, q);
}

double Option_Price::BSM_Delta() {
  double S = getCurrentPrice();
  double K = getStrikePrice();
  double r = getRiskFreeRate();
  double T = getTimeToMaturity();
  double sigma = getVolatility();

  double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));

  if (flag == 'c' || flag == 'C') {
    return normalCDF(d1);
  } else {
    return normalCDF(d1) - 1.0;
  }
}

double Option_Price::Binomial_Delta() {
  double S = getCurrentPrice();
  double K = getStrikePrice();
  double T = getTimeToMaturity();
  double sigma = getVolatility();
  double r = getRiskFreeRate();

  int n = 100;
  double dt = T / n;
  double u = exp(sigma * sqrt(dt));
  double d = 1.0 / u;
  double p = (exp(r * dt) - d) / (u - d);
  double q = 1.0 - p;

  double priceUp = binomialPrice(n - 1, u, d, p, q);
  double priceDown = binomialPrice(n - 1, u, d, p, q);

  double S_up = S * u;
  double S_down = S * d;

  double payoffUp, payoffDown;
  if (flag == 'c' || flag == 'C') {
    payoffUp = std::max(S_up - K, 0.0);
    payoffDown = std::max(S_down - K, 0.0);
  } else if (flag == 'p' || flag == 'P') {
    payoffUp = std::max(K - S_up, 0.0);
    payoffDown = std::max(K - S_down, 0.0);
  } else {
    throw std::invalid_argument("Invalid option type");
  }

  return (payoffUp - payoffDown) / (S_up - S_down);
}
