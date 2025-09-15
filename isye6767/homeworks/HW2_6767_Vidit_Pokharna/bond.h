#ifndef BOND_H
#define BOND_H

#include <string>
#include <vector>

class Bond
{
private:
    std::string expiration_date_;
    double frequency_;
    double coupon_rate_;

public:
    Bond();
    ~Bond();
    Bond(const Bond &other);
    Bond(const std::string &expiration_date, double frequency, double coupon_rate);
    std::string ToString() const;

    std::string expiration_date() const { return expiration_date_; }
    double frequency() const { return frequency_; }
    double coupon_rate() const { return coupon_rate_; }

    double PriceFlat(double r,
                     double T,
                     double tN,
                     double face = 100.0) const;

    double Price(const std::vector<double> &zero_times,
                 const std::vector<double> &zero_rates,
                 double T,
                 double tN,
                 double face = 100.0) const;

    void set_expiration_date(const std::string &d) { expiration_date_ = d; }
    void set_frequency(double f) { frequency_ = f; }
    void set_coupon_rate(double c) { coupon_rate_ = c; }
};

#endif
