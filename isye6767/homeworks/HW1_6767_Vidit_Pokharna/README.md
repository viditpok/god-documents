# HW1 - ISyE6767

## Files
- `homework_1.cpp` — C++ source code
- `hw1_H.15_Baa_Data.csv` — data file
- `report.pdf` — brief report

## What the Program Does
- Loads the Baa corporate bond yields CSV
- Detects the date column from the header
- Computes the average Baa yield across the dataset
- Repeatedly prompts the user for a `yyyy-mm` date until EOF:
  - If the date exists: prints that month’s yield, the dataset average, and the difference
  - If the date does not exist: prints a “not found” message

### Key Functions
- `average(...)`: Computes mean of the loaded yields
- `find_rate(...)`: Returns the yield for an input `yyyy-mm` or NaN if not found

## How to Compile
Use a C++11 (or newer) compiler:
```bash
g++ -std=gnu++11 -O2 -Wall -Wextra -o hw1 homework_1.cpp
