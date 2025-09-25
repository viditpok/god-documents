#ifndef PRICING_METHOD_H
#define PRICING_METHOD_H

class Pricing_Method {
public:
  virtual double BSM_Pricer() = 0;
  virtual double Binomial_Pricer() = 0;
  virtual ~Pricing_Method() = default;
};

#endif
