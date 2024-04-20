# Setting the working directory and loading required libraries
setwd("/Users/viditpokharna/Downloads")
library(dplyr)
library(ggplot2)

# Reading wind ordinance data and filtering for years 2001 onwards
wind_ordinance_data <- read.csv("wind_ordinance_main.csv")
wind_ordinance_data <- subset(wind_ordinance_data, ordinance_year >= 2001)

# Counting ordinances by state and identifying the top and bottom three states
ordinance_count <- wind_ordinance_data %>%
  count(State) %>%
  arrange(desc(n))
top_3_states <- head(ordinance_count, 3)
bottom_3_states <- tail(ordinance_count, 3)

# Grouping ordinance data by year and calculating the annual and cumulative counts
yearly_ordinance_data <- wind_ordinance_data %>%
  group_by(ordinance_year) %>%
  summarise(total_ordinances = n()) %>%
  mutate(cumulative_ordinances = cumsum(total_ordinances))

# Plotting yearly and cumulative wind ordinances over time
ggplot(yearly_ordinance_data) +
  geom_line(aes(x = ordinance_year, y = total_ordinances), color = "blue") +
  geom_line(aes(x = ordinance_year, y = cumulative_ordinances), color = "red") +
  labs(title = "Wind Ordinances Over Time", x = "Year", y = "Number of Ordinances")
