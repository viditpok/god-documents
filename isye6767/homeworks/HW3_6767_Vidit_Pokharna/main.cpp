#include "Option_Price.h"
#include "test.h"
#include <iomanip>
#include <iostream>

int main() {
  std::cout << "=== Option Pricing Calculator ===" << std::endl;
  std::cout << "Using Black-Scholes-Merton and Binomial Lattice Methods"
            << std::endl;
  std::cout << std::endl;

  double strike, spot, rate, time, vol;
  char optionFlag;

  std::cout << "Please enter the option parameters:" << std::endl;
  std::cout << "Strike Price (K): ";
  std::cin >> strike;

  std::cout << "Current Price of Underlying (S): ";
  std::cin >> spot;

  std::cout << "Risk-free Rate (r): ";
  std::cin >> rate;

  std::cout << "Time to Maturity (T): ";
  std::cin >> time;

  std::cout << "Volatility (sigma): ";
  std::cin >> vol;

  std::cout << "Option Type (c/C for Call, p/P for Put): ";
  std::cin >> optionFlag;

  while (optionFlag != 'c' && optionFlag != 'C' && optionFlag != 'p' &&
         optionFlag != 'P') {
    std::cout
        << "Invalid option type. Please enter c/C for Call or p/P for Put: ";
    std::cin >> optionFlag;
  }

  std::cout << std::endl;
  std::cout << "=== Option Parameters ===" << std::endl;
  std::cout << "Strike Price (K): " << strike << std::endl;
  std::cout << "Current Price (S): " << spot << std::endl;
  std::cout << "Risk-free Rate (r): " << rate << std::endl;
  std::cout << "Time to Maturity (T): " << time << std::endl;
  std::cout << "Volatility (sigma): " << vol << std::endl;
  std::cout << "Option Type: "
            << (optionFlag == 'c' || optionFlag == 'C' ? "Call" : "Put")
            << std::endl;
  std::cout << std::endl;

  Option_Price option(strike, spot, rate, time, vol, optionFlag);

  double bsmPrice = option.BSM_Pricer();
  double binomialPrice = option.Binomial_Pricer();

  double bsmDelta = option.BSM_Delta();
  double binomialDelta = option.Binomial_Delta();

  std::cout << "=== Pricing Results ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Black-Scholes-Merton Price: " << bsmPrice << std::endl;
  std::cout << "Binomial Lattice Price:     " << binomialPrice << std::endl;
  std::cout << "Price Difference:           "
            << std::abs(bsmPrice - binomialPrice) << std::endl;
  std::cout << std::endl;

  std::cout << "=== Delta Values ===" << std::endl;
  std::cout << "Black-Scholes-Merton Delta: " << bsmDelta << std::endl;
  std::cout << "Binomial Lattice Delta:     " << binomialDelta << std::endl;
  std::cout << "Delta Difference:           "
            << std::abs(bsmDelta - binomialDelta) << std::endl;
  std::cout << std::endl;

  std::cout << "=== Running Unit Tests ===" << std::endl;
  runTests();

  return 0;
}
