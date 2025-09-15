#include "bond.h"
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace
{
    static std::string trim_double(double x)
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(6) << x;
        std::string s = oss.str();
        while (!s.empty() and s.back() == '0')
            s.pop_back();
        if (!s.empty() and s.back() == '.')
            s.pop_back();
        if (s.empty())
            s = "0";
        return s;
    }
}

Bond::Bond() : expiration_date_("0"), frequency_(0.0), coupon_rate_(0.0) {}

Bond::~Bond() = default;

Bond::Bond(const Bond &other)
    : expiration_date_(other.expiration_date_),
      frequency_(other.frequency_),
      coupon_rate_(other.coupon_rate_) {}

Bond::Bond(const std::string &expiration_date, double frequency, double coupon_rate)
    : expiration_date_(expiration_date), frequency_(frequency), coupon_rate_(coupon_rate) {}

std::string Bond::ToString() const
{
    std::ostringstream oss;
    oss << "Bond ("
        << expiration_date_ << ", "
        << trim_double(frequency_) << ", "
        << trim_double(coupon_rate_) << ")";
    return oss.str();
}

namespace
{

    static double nearest_rate(double t,
                               const std::vector<double> &times,
                               const std::vector<double> &rates)
    {
        if (times.empty() || rates.empty() || times.size() != rates.size())
            return 0.0;
        size_t best = 0;
        double best_diff = std::abs(times[0] - t);
        for (size_t i = 1; i < times.size(); ++i)
        {
            double d = std::abs(times[i] - t);
            if (d < best_diff)
            {
                best_diff = d;
                best = i;
            }
        }
        return rates[best];
    }
}

static std::vector<double> build_times(double T, double tN, double period)
{
    std::vector<double> times;
    const double eps = 1e-9;
    if (T <= 0.0)
        return times;
    if (tN > 0.0 && tN < T + eps)
    {
        for (double t = tN; t + eps < T; t += period)
            times.push_back(t);
    }
    times.push_back(T);
    return times;
}

template <class RateFn>
static double price_internal(double face,
                             double coupon_rate,
                             double period,
                             double T,
                             double tN,
                             RateFn r_of_t)
{

    int ppy = (period > 0.0) ? static_cast<int>(std::lround(1.0 / period)) : 0;
    if (ppy <= 0)
        return std::numeric_limits<double>::quiet_NaN();

    if (T <= 1e-12)
    {
        double last_coupon = face * coupon_rate / static_cast<double>(ppy);
        return face + last_coupon;
    }

    const double full_coupon = face * coupon_rate / static_cast<double>(ppy);
    const double first_coupon = full_coupon * (tN / period);

    auto times = build_times(T, tN, period);
    if (times.empty())
        return std::numeric_limits<double>::quiet_NaN();

    double pv = 0.0;
    for (size_t k = 0; k < times.size(); ++k)
    {
        double t = times[k];

        double cf = (k == 0 ? first_coupon : full_coupon);
        if (k == times.size() - 1)
            cf += face;

        double rk = r_of_t(t);
        pv += cf * std::exp(-rk * t);
    }
    return pv;
}

double Bond::PriceFlat(double r, double T, double tN, double face) const
{

    const double period = frequency_ > 0.0 ? frequency_ : 0.0;
    return price_internal(face, coupon_rate_, period, T, tN,
                          [r](double)
                          { return r; });
}

double Bond::Price(const std::vector<double> &zero_times,
                   const std::vector<double> &zero_rates,
                   double T,
                   double tN,
                   double face) const
{
    const double period = frequency_ > 0.0 ? frequency_ : 0.0;
    return price_internal(face, coupon_rate_, period, T, tN,
                          [&](double t)
                          { return nearest_rate(t, zero_times, zero_rates); });
}
