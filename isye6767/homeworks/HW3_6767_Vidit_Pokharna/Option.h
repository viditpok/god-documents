#ifndef OPTION_H
#define OPTION_H

class Option {
private:
  double K;
  double S;
  double r;
  double T;
  double sigma;
  void init();

public:
  Option();
  Option(double strike, double spot, double rate, double time, double vol);
  ~Option();
  double getStrikePrice() const;
  double getCurrentPrice() const;
  double getRiskFreeRate() const;
  double getTimeToMaturity() const;
  double getVolatility() const;
};

#endif
