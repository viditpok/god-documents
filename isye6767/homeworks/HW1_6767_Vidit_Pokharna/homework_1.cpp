#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>

using namespace std;

double average(vector<double> v)
{
   if (v.empty())
      return numeric_limits<double>::quiet_NaN();
   long double sum = 0.0L;
   for (double x : v)
      sum += x;
   return static_cast<double>(sum / v.size());
}

double find_rate(vector<double> rate_vec, vector<string> date_vec, string date)
{
   for (size_t i = 0; i < date_vec.size(); ++i)
   {
      if (date_vec[i] == date)
         return rate_vec[i];
   }
   return numeric_limits<double>::quiet_NaN();
}

static inline string trim(const string &s)
{
   size_t b = 0, e = s.size();
   while (b < e && isspace(static_cast<unsigned char>(s[b])))
      ++b;
   while (e > b && isspace(static_cast<unsigned char>(s[e - 1])))
      --e;
   return s.substr(b, e - b);
}

int main()
{
   vector<double> rates;
   vector<string> dates;
   ifstream infile("./hw1_H.15_Baa_Data.csv");

   if (!infile.is_open())
   {
      cerr << "Error: cannot open ./hw1_H.15_Baa_Data.csv\n";
      return 1;
   }

   string header;
   if (!getline(infile, header))
   {
      cerr << "Error: empty CSV or unreadable header.\n";
      return 1;
   }

   vector<string> cols;
   {
      string cell;
      stringstream ss(header);
      while (getline(ss, cell, ','))
         cols.push_back(trim(cell));
   }

   int date_col = -1, rate_col = -1;
   for (size_t i = 0; i < cols.size(); ++i)
   {
      string c = cols[i];
      transform(c.begin(), c.end(), c.begin(), ::toupper);
      if (c == "DATE")
         date_col = static_cast<int>(i);
      if (c == "BAA" || c == "BAA YIELD" || c == "BAA_YIELD")
         rate_col = static_cast<int>(i);
   }

   if (date_col < 0 && !cols.empty())
      date_col = 0;
   if (rate_col < 0 && static_cast<int>(cols.size()) >= 2)
      rate_col = 1;

   if (date_col < 0 || rate_col < 0)
   {
      cerr << "Error: could not locate DATE and BAA columns from header.\n";
      return 1;
   }

   string line;
   while (getline(infile, line))
   {
      if (line.empty())
         continue;
      vector<string> fields;
      {
         string cell;
         stringstream ss(line);
         while (getline(ss, cell, ','))
            fields.push_back(trim(cell));
      }
      if (static_cast<int>(fields.size()) <= max(date_col, rate_col))
         continue;

      string d = fields[date_col];
      string rstr = fields[rate_col];

      if (rstr.empty() || rstr == ".")
         continue;

      try
      {
         double r = stod(rstr);
         dates.push_back(d);
         rates.push_back(r);
      }
      catch (...)
      {
         continue;
      }
   }
   infile.close();

   double avg_rate = average(rates);

   cout.setf(ios::fixed);
   cout.precision(4);

   if (rates.empty())
   {
      cout << "No data rows with valid rates were found. Exiting.\n";
      return 0;
   }

   cout << "Data loaded: " << rates.size() << " rows.\n";
   cout << "Enter dates in yyyy-mm (EOF to quit). Example: 2010-07\n";

   string query;
   while (true)
   {
      cout << "\nDate> ";
      if (!getline(cin, query))
      {
         cout << "\nGoodbye.\n";
         break;
      }
      query = trim(query);
      if (query.empty())
      {
         cout << "Empty input; please enter a date like yyyy-mm.\n";
         continue;
      }

      double r = find_rate(rates, dates, query);
      if (std::isnan(r))
      {
         cout << "Date " << query << " not found in data. Please try another date.\n";
         continue;
      }

      double diff = r - avg_rate;
      cout << "Baa rate for " << query << " = " << r << "%\n";
      cout << "Average rate across dataset = " << avg_rate << "%\n";
      cout << "Difference (date - average) = " << diff << " percentage points\n";
   }

   return 0.0; // program end
}
