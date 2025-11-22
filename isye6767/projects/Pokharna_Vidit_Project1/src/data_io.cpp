#include "delta_hedging/data_io.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "delta_hedging/black_scholes.h"

namespace delta_hedging
{

    namespace
    {

        struct Date
        {
            int year;
            int month;
            int day;
        };

        Date parse_date(const std::string &value)
        {
            if (value.size() != 10 || value[4] != '-' || value[7] != '-')
            {
                throw std::invalid_argument("Invalid date format: " + value);
            }
            return Date{std::stoi(value.substr(0, 4)), std::stoi(value.substr(5, 2)), std::stoi(value.substr(8, 2))};
        }

        std::string date_key(const Date &date)
        {
            std::ostringstream oss;
            oss << date.year << '-';
            if (date.month < 10)
            {
                oss << '0';
            }
            oss << date.month << '-';
            if (date.day < 10)
            {
                oss << '0';
            }
            oss << date.day;
            return oss.str();
        }

        bool operator<(const Date &lhs, const Date &rhs)
        {
            if (lhs.year != rhs.year)
            {
                return lhs.year < rhs.year;
            }
            if (lhs.month != rhs.month)
            {
                return lhs.month < rhs.month;
            }
            return lhs.day < rhs.day;
        }

        bool operator<=(const Date &lhs, const Date &rhs)
        {
            return !(rhs < lhs);
        }

        bool operator==(const Date &lhs, const Date &rhs)
        {
            return lhs.year == rhs.year && lhs.month == rhs.month && lhs.day == rhs.day;
        }

        Date advance_one_day(const Date &date)
        {
            static const int days_per_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            int day = date.day + 1;
            int month = date.month;
            int year = date.year;
            int days_this_month = days_per_month[month - 1];
            bool leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if (month == 2 && leap)
            {
                days_this_month = 29;
            }
            if (day > days_this_month)
            {
                day = 1;
                month += 1;
                if (month > 12)
                {
                    month = 1;
                    year += 1;
                }
            }
            return Date{year, month, day};
        }

        bool is_business_day(const Date &date)
        {
            int y = date.year;
            int m = date.month;
            int d = date.day;
            if (m < 3)
            {
                m += 12;
                y -= 1;
            }
            int K = y % 100;
            int J = y / 100;
            int h = (d + 13 * (m + 1) / 5 + K + K / 4 + J / 4 + 5 * J) % 7;
            int day_of_week = ((h + 5) % 7) + 1;
            return day_of_week <= 5;
        }

        int business_days_between(const Date &start, const Date &end)
        {
            if (end < start)
            {
                return 0;
            }
            int count = 0;
            Date current = start;
            while (current < end)
            {
                current = advance_one_day(current);
                if (is_business_day(current))
                {
                    ++count;
                }
            }
            return count;
        }

        double to_double(const std::string &value)
        {
            return std::stod(value);
        }

        std::unordered_map<std::string, double> read_rate_file(const std::string &path)
        {
            std::unordered_map<std::string, double> rates;
            std::ifstream in(path);
            if (!in)
            {
                throw std::runtime_error("Unable to open rate file: " + path);
            }
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line))
            {
                if (line.empty())
                {
                    continue;
                }
                std::istringstream iss(line);
                std::string date, rate;
                if (!std::getline(iss, date, ','))
                {
                    continue;
                }
                if (!std::getline(iss, rate))
                {
                    continue;
                }
                rates.emplace(date, to_double(rate) / 100.0);
            }
            return rates;
        }

        std::unordered_map<std::string, double> read_stock_file(const std::string &path)
        {
            std::unordered_map<std::string, double> stocks;
            std::ifstream in(path);
            if (!in)
            {
                throw std::runtime_error("Unable to open stock file: " + path);
            }
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line))
            {
                if (line.empty())
                {
                    continue;
                }
                std::istringstream iss(line);
                std::string date, close;
                if (!std::getline(iss, date, ','))
                {
                    continue;
                }
                if (!std::getline(iss, close))
                {
                    continue;
                }
                stocks.emplace(date, to_double(close));
            }
            return stocks;
        }

        struct OptionRow
        {
            std::string date;
            std::string exdate;
            std::string cp_flag;
            double strike;
            double price;
        };

        std::vector<OptionRow> read_option_file(const std::string &path)
        {
            std::vector<OptionRow> rows;
            std::ifstream in(path);
            if (!in)
            {
                throw std::runtime_error("Unable to open option file: " + path);
            }
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line))
            {
                if (line.empty())
                {
                    continue;
                }
                std::istringstream iss(line);
                std::string date, exdate, cp_flag, strike_str, bid_str, ask_str;
                if (!std::getline(iss, date, ','))
                    continue;
                if (!std::getline(iss, exdate, ','))
                    continue;
                if (!std::getline(iss, cp_flag, ','))
                    continue;
                if (!std::getline(iss, strike_str, ','))
                    continue;
                if (!std::getline(iss, bid_str, ','))
                    continue;
                if (!std::getline(iss, ask_str, ','))
                    continue;
                double strike = to_double(strike_str);
                double price = 0.5 * (to_double(bid_str) + to_double(ask_str));
                rows.push_back(OptionRow{date, exdate, cp_flag, strike, price});
            }
            return rows;
        }

        MarketRecord make_record(const Date &day, double stock_price, double option_price, double rate, double strike,
                                 const Date &expiry)
        {
            const double tau = business_days_between(day, expiry) / 252.0;
            double implied = 0.0;
            if (tau > 0.0 && option_price > 0.0)
            {
                try
                {
                    implied = implied_volatility(option_price, stock_price, strike, rate, tau);
                }
                catch (const std::exception &)
                {
                    implied = 0.0;
                }
            }
            double delta = (tau > 0.0) ? black_scholes_delta(stock_price, strike, rate, implied, tau)
                                       : (stock_price > strike ? 1.0 : 0.0);
            return MarketRecord{day.year, day.month, day.day, stock_price, option_price, rate, tau, implied, delta};
        }

        Date parse_date_string(const std::string &date)
        {
            return parse_date(date);
        }

    }

    std::vector<MarketRecord> load_market_window(const Task2Config &config)
    {
        const std::string rate_file = config.data_dir + "/interest.csv";
        const std::string stock_file = config.data_dir + "/sec_GOOG.csv";
        const std::string option_file = config.data_dir + "/op_GOOG.csv";

        auto rate_map = read_rate_file(rate_file);
        auto stock_map = read_stock_file(stock_file);
        auto option_rows = read_option_file(option_file);
        Date expiry = parse_date_string(config.expiry);
        Date start = parse_date_string(config.start_date);
        Date end = parse_date_string(config.end_date);

        std::vector<MarketRecord> records;
        for (const auto &row : option_rows)
        {
            if (row.cp_flag != config.cp_flag)
            {
                continue;
            }
            if (std::abs(row.strike - config.strike) > 1e-8)
            {
                continue;
            }
            if (row.exdate != config.expiry)
            {
                continue;
            }
            const auto stock_it = stock_map.find(row.date);
            if (stock_it == stock_map.end())
            {
                continue;
            }
            Date day = parse_date_string(row.date);
            if (!(start <= day && day <= end))
            {
                continue;
            }
            const auto rate_it = rate_map.find(row.date);
            double rate = (rate_it != rate_map.end()) ? rate_it->second : 0.0;
            MarketRecord record = make_record(day, stock_it->second, row.price, rate, config.strike, expiry);
            records.push_back(record);
        }
        std::sort(records.begin(), records.end(), [](const MarketRecord &lhs, const MarketRecord &rhs)
                  {
        if (lhs.year != rhs.year) return lhs.year < rhs.year;
        if (lhs.month != rhs.month) return lhs.month < rhs.month;
        return lhs.day < rhs.day; });
        return records;
    }

    int business_days_between_ymd(int y1, int m1, int d1, int y2, int m2, int d2)
    {
        Date start{y1, m1, d1};
        Date end{y2, m2, d2};
        return business_days_between(start, end);
    }

}
