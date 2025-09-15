#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cctype>
#include "bond.h"

static bool approx_equal(double a, double b, double tol = 1e-8)
{
    return std::fabs(a - b) <= tol * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

static void report(const std::string &name, bool ok)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << std::endl;
}

static bool test_ToString_exact()
{
    Bond b("01/01/2020", 0.5, 0.07);
    return b.ToString() == std::string("Bond (01/01/2020, 0.5, 0.07)");
}

static bool test_Default_ctor_zeroes()
{
    Bond b;
    return b.ToString() == std::string("Bond (0, 0, 0)");
}

static bool test_Flat_monotone()
{
    Bond b("11/19/2035", 0.5, 0.07);
    double T = 5.0, tN = 0.5;
    double p3 = b.PriceFlat(0.03, T, tN);
    double p5 = b.PriceFlat(0.05, T, tN);
    double p7 = b.PriceFlat(0.07, T, tN);
    return (p3 > p5) && (p5 > p7);
}

static bool test_Curve_equals_flat()
{
    Bond b("11/19/2035", 0.5, 0.07);
    double T = 5.0, tN = 0.5;
    std::vector<double> times{0.5, 1, 2, 3, 4, 5};
    std::vector<double> rates(times.size(), 0.05);
    double pf = b.PriceFlat(0.05, T, tN);
    double pc = b.Price(times, rates, T, tN);
    return approx_equal(pf, pc, 1e-6);
}

static bool test_First_coupon_proration()
{
    Bond b("12/31/2025", 1.0, 0.05);
    double T = 1.0, tN = 0.25, r = 0.0;
    double price = b.PriceFlat(r, T, tN);
    return approx_equal(price, 106.25, 1e-10);
}

static std::string lower(std::string s)
{
    for (auto &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static bool load_zero_curve_csv(const std::string &path,
                                std::vector<double> &times,
                                std::vector<double> &rates)
{
    std::ifstream fin(path);
    if (!fin)
        return false;

    std::string line;
    if (!std::getline(fin, line))
        return false;

    std::vector<std::string> cols;
    {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            cols.push_back(tok);
    }
    int i_ttm = -1, i_rate = -1;
    for (int i = 0; i < (int)cols.size(); ++i)
    {
        std::string h = lower(cols[i]);
        if (i_ttm < 0 && (h.find("ttm") != std::string::npos ||
                          h.find("time to maturity") != std::string::npos))
            i_ttm = i;
        if (i_rate < 0 && (h.find("interest rate") != std::string::npos ||
                           h.find("rate") != std::string::npos ||
                           h.find("yield") != std::string::npos ||
                           h.find("zero") != std::string::npos))
            i_rate = i;
    }
    if (i_ttm < 0 || i_rate < 0)
    {
        i_ttm = 0;
        i_rate = 1;
    }

    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> row;
        while (std::getline(ss, tok, ','))
            row.push_back(tok);
        if ((int)row.size() <= std::max(i_ttm, i_rate))
            continue;
        try
        {
            double t = std::stod(row[i_ttm]);
            double r = std::stod(row[i_rate]);
            times.push_back(t);
            rates.push_back(r);
        }
        catch (...)
        {
        }
    }
    return !times.empty() && times.size() == rates.size();
}

static double nn_rate(double T,
                      const std::vector<double> &times,
                      const std::vector<double> &rates)
{
    if (times.empty())
        return 0.0;
    size_t best = 0;
    double bestd = std::fabs(times[0] - T);
    for (size_t i = 1; i < times.size(); ++i)
    {
        double d = std::fabs(times[i] - T);
        if (d < bestd)
        {
            bestd = d;
            best = i;
        }
    }
    return rates[best];
}

static long long days_from_civil(int y, unsigned m, unsigned d)
{
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}
static double yearfrac_act_365(int y1, unsigned m1, unsigned d1,
                               int y2, unsigned m2, unsigned d2)
{
    long long n = days_from_civil(y2, m2, d2) - days_from_civil(y1, m1, d1);
    return static_cast<double>(n) / 365.0;
}

int main()
{
    std::cout << "\n--- Demo ---\n";
    Bond default_bond;
    std::cout << default_bond.ToString() << std::endl;
    Bond example("11/19/2035", 0.5, 0.07);
    std::cout << example.ToString() << std::endl;

    std::cout << "\n--- Bond Pricing Unit Tests ---\n";
    bool success = true;

    bool ok = test_ToString_exact();
    report("ToString exact", ok);
    success &= ok;
    ok = test_Default_ctor_zeroes();
    report("Default ctor zeroes", ok);
    success &= ok;
    ok = test_Flat_monotone();
    report("Flat-rate monotone", ok);
    success &= ok;
    ok = test_Curve_equals_flat();
    report("Curve equals flat TS", ok);
    success &= ok;
    ok = test_First_coupon_proration();
    report("First coupon proration", ok);
    success &= ok;

    std::cout << (success ? "\nAll tests passed\n" : "\nSome tests failed\n");

    std::cout << "\n--- Investment Analysis Using Pricing Data ---\n";

    std::vector<double> zt, zr;
    if (!load_zero_curve_csv("Bond_Ex3.csv", zt, zr))
    {
        std::cerr << "[ERROR] could not read Bond_Ex3.csv\n";
        return 1;
    }

    Bond underlying("01/01/2020", 0.5, 0.05);
    const double face = 100.0;
    double prices[5];
    double ttms[5] = {4.0, 3.0, 2.0, 1.0, 0.0};
    double tns[5] = {0.5, 0.5, 0.5, 0.5, 0.0};
    for (int i = 0; i < 5; ++i)
    {
        prices[i] = underlying.Price(zt, zr, ttms[i], tns[i], face);
        std::cout << "Jan 1, " << (2016 + i)
                  << "  TTM=" << ttms[i]
                  << "  Price=" << prices[i] << "\n";
    }

    double avg = 0.0;
    for (double p : prices)
        avg += p;
    avg /= 5.0;
    std::cout << "Average price (2016-2020): " << avg << "\n";

    double Tpay = yearfrac_act_365(2015, 8, 3, 2020, 12, 31);
    double r_disc = nn_rate(Tpay, zt, zr);
    double df = std::exp(-r_disc * Tpay);
    double pv_at_purchase = avg * df;

    const double purchase_price = 98.0;
    double npv = pv_at_purchase - purchase_price;

    std::cout << "Year fraction 2015-08-03 -> 2020-12-31 (ACT/365): " << Tpay << "\n";
    std::cout << "Discount rate (nearest-neighbor at T=" << Tpay << "): " << r_disc << "\n";
    std::cout << "Discount factor: " << df << "\n";
    std::cout << "PV at purchase: " << pv_at_purchase << "\n";
    std::cout << "NPV vs $98: " << npv << "  =>  "
              << (npv > 0 ? "Good investment" : "Bad investment") << "\n";

    return success ? 0 : 1;
}
