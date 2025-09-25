#ifndef OPTION_PRICE_H
#define OPTION_PRICE_H

#include "Option.h"
#include "pricing_method.h"
#include <string>

class Option_Price : public Option, public Pricing_Method {
private:
  char flag;

  double normalCDF(double x);
  double normalPDF(double x);

  double binomialCoeff(int n, int k);
  double binomialPrice(int n, double u, double d, double p, double q);

public:
  Option_Price(double strike, double spot, double rate, double time, double vol,
               char optionFlag);
  ~Option_Price();

  char getFlag() const;

  double BSM_Pricer() override;
  double Binomial_Pricer() override;

  double BSM_Delta();
  double Binomial_Delta();
};

#endif
