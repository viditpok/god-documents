#ifndef TEST_H
#define TEST_H

#include "Option_Price.h"
#include <cmath>
#include <iomanip>
#include <iostream>

void testOptionClass();
void testBSMPricing();
void testBinomialPricing();
void testDeltaCalculation();
void testEdgeCases();
void runTests();

void testOptionClass() {
  std::cout << "Testing Option Class..." << std::endl;

  Option defaultOption;
  std::cout << "  Default constructor - Strike: "
            << defaultOption.getStrikePrice()
            << ", Spot: " << defaultOption.getCurrentPrice() << std::endl;

  Option customOption(105.0, 100.0, 0.05, 0.5, 0.25);
  std::cout << "  Custom constructor - Strike: "
            << customOption.getStrikePrice()
            << ", Spot: " << customOption.getCurrentPrice() << std::endl;

  std::cout << "  Option Class tests passed!" << std::endl;
  std::cout << std::endl;
}

void testBSMPricing() {
  std::cout << "Testing Black-Scholes-Merton Pricing..." << std::endl;

  Option_Price atmCall(100.0, 100.0, 0.05, 1.0, 0.2, 'c');
  double bsmPrice1 = atmCall.BSM_Pricer();
  std::cout << "  ATM Call (S=100, K=100): " << bsmPrice1 << std::endl;

  Option_Price otmPut(90.0, 100.0, 0.05, 1.0, 0.2, 'p');
  double bsmPrice2 = otmPut.BSM_Pricer();
  std::cout << "  OTM Put (S=100, K=90): " << bsmPrice2 << std::endl;

  Option_Price itmCall(80.0, 100.0, 0.05, 1.0, 0.2, 'c');
  double bsmPrice3 = itmCall.BSM_Pricer();
  std::cout << "  ITM Call (S=100, K=80): " << bsmPrice3 << std::endl;

  std::cout << "  BSM Pricing tests completed!" << std::endl;
  std::cout << std::endl;
}

void testBinomialPricing() {
  std::cout << "Testing Binomial Lattice Pricing..." << std::endl;

  Option_Price atmCall(100.0, 100.0, 0.05, 1.0, 0.2, 'c');
  double binomialPrice1 = atmCall.Binomial_Pricer();
  std::cout << "  ATM Call (S=100, K=100): " << binomialPrice1 << std::endl;

  Option_Price otmPut(90.0, 100.0, 0.05, 1.0, 0.2, 'p');
  double binomialPrice2 = otmPut.Binomial_Pricer();
  std::cout << "  OTM Put (S=100, K=90): " << binomialPrice2 << std::endl;

  Option_Price itmCall(80.0, 100.0, 0.05, 1.0, 0.2, 'c');
  double binomialPrice3 = itmCall.Binomial_Pricer();
  std::cout << "  ITM Call (S=100, K=80): " << binomialPrice3 << std::endl;

  std::cout << "  Binomial Pricing tests completed!" << std::endl;
  std::cout << std::endl;
}

void testDeltaCalculation() {
  std::cout << "Testing Delta Calculations..." << std::endl;

  Option_Price testOption(100.0, 100.0, 0.05, 1.0, 0.2, 'c');

  double bsmDelta = testOption.BSM_Delta();
  double binomialDelta = testOption.Binomial_Delta();

  std::cout << "  BSM Delta: " << bsmDelta << std::endl;
  std::cout << "  Binomial Delta: " << binomialDelta << std::endl;
  std::cout << "  Delta Difference: " << std::abs(bsmDelta - binomialDelta)
            << std::endl;

  std::cout << "  Delta calculation tests completed!" << std::endl;
  std::cout << std::endl;
}

void testEdgeCases() {
  std::cout << "Testing Edge Cases..." << std::endl;

  Option_Price shortTerm(100.0, 100.0, 0.05, 0.001, 0.2, 'c');
  double shortTermBSM = shortTerm.BSM_Pricer();
  double shortTermBin = shortTerm.Binomial_Pricer();
  std::cout << "  Very short term (T=0.001): BSM=" << shortTermBSM
            << ", Binomial=" << shortTermBin << std::endl;

  Option_Price highVol(100.0, 100.0, 0.05, 1.0, 1.0, 'c');
  double highVolBSM = highVol.BSM_Pricer();
  double highVolBin = highVol.Binomial_Pricer();
  std::cout << "  High volatility (sigma=1.0): BSM=" << highVolBSM
            << ", Binomial=" << highVolBin << std::endl;

  std::cout << "  Edge case tests completed!" << std::endl;
  std::cout << std::endl;
}

void runTests() {
  std::cout << std::fixed << std::setprecision(6);

  testOptionClass();
  testBSMPricing();
  testBinomialPricing();
  testDeltaCalculation();
  testEdgeCases();

  std::cout << "All unit tests completed!" << std::endl;
  std::cout << std::endl;
}

#endif
