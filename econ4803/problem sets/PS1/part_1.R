# Set the working directory and load necessary libraries
setwd("/Users/viditpokharna/Downloads")
library(dplyr)
library(ggplot2)
library(knitr)

# Load and preprocess the dataset
data <- read.csv("uswtdb_v6_1_20231128.csv")
data <- data %>%
  filter(!is.na(p_year) & !is.na(p_cap) & !is.na(t_cap) & !is.na(t_hh) & !is.na(t_rd) & !is.na(xlong) & !is.na(ylat),
         p_year >= 2001,
         !t_state %in% c("AK", "HI"))

# Generate a summary table of wind turbine statistics
summary_table <- data %>%
  mutate(t_cap_MW = t_cap / 1000) %>%
  summarise(
    Mean_Capacity_MW = mean(t_cap_MW, na.rm = TRUE),
    SD_Capacity_MW = sd(t_cap_MW, na.rm = TRUE),
    Min_Capacity_MW = min(t_cap_MW, na.rm = TRUE),
    Max_Capacity_MW = max(t_cap_MW, na.rm = TRUE),
    Mean_Hub_Height = mean(t_hh, na.rm = TRUE),
    SD_Hub_Height = sd(t_hh, na.rm = TRUE),
    Min_Hub_Height = min(t_hh, na.rm = TRUE),
    Max_Hub_Height = max(t_hh, na.rm = TRUE),
    Mean_Rotor_Diameter = mean(t_rd, na.rm = TRUE),
    SD_Rotor_Diameter = sd(t_rd, na.rm = TRUE),
    Min_Rotor_Diameter = min(t_rd, na.rm = TRUE),
    Max_Rotor_Diameter = max(t_rd, na.rm = TRUE),
    Total_Turbines = n()
  )

# Aggregate data at the project level
aggregated_data <- data %>%
  mutate(t_cap_MW = t_cap / 1000) %>%
  group_by(eia_id) %>%
  summarise(
    Mean_Project_Capacity_MW = mean(p_cap, na.rm = TRUE),
    Operating_Year = first(p_year),
    Total_Turbines = n(),
    Mean_Turbine_Capacity_MW = mean(t_cap_MW, na.rm = TRUE),
    Mean_Hub_Height = mean(t_hh, na.rm = TRUE),
    Mean_Rotor_Diameter = mean(t_rd, na.rm = TRUE),
    State = first(t_state),
    County = first(t_county)
  )

# Generate a project summary table
project_summary_table <- aggregated_data %>%
  summarise(
    Mean_Project_Capacity = mean(Mean_Project_Capacity_MW, na.rm = TRUE),
    SD_Project_Capacity = sd(Mean_Project_Capacity_MW, na.rm = TRUE),
    Min_Project_Capacity = min(Mean_Project_Capacity_MW, na.rm = TRUE),
    Max_Project_Capacity = max(Mean_Project_Capacity_MW, na.rm = TRUE),
    Mean_Turbines_Per_Project = mean(Total_Turbines, na.rm = TRUE),
    SD_Turbines_Per_Project = sd(Total_Turbines, na.rm = TRUE),
    Min_Turbines_Per_Project = min(Total_Turbines, na.rm = TRUE),
    Max_Turbines_Per_Project = max(Total_Turbines, na.rm = TRUE),
    Mean_Hub_Height = mean(Mean_Hub_Height, na.rm = TRUE),
    SD_Hub_Height = sd(Mean_Hub_Height, na.rm = TRUE),
    Min_Hub_Height = min(Mean_Hub_Height, na.rm = TRUE),
    Max_Hub_Height = max(Mean_Hub_Height, na.rm = TRUE),
    Mean_Rotor_Diameter = mean(Mean_Rotor_Diameter, na.rm = TRUE),
    SD_Rotor_Diameter = sd(Mean_Rotor_Diameter, na.rm = TRUE),
    Min_Rotor_Diameter = min(Mean_Rotor_Diameter, na.rm = TRUE),
    Max_Rotor_Diameter = max(Mean_Rotor_Diameter, na.rm = TRUE),
    Total_Projects = n()
  )

# Plot trends in hub height and rotor diameter over time
ggplot(aggregated_data, aes(x = Operating_Year, y = Mean_Hub_Height)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Evolution of Mean Hub-Height Over the Years", 
       x = "Year", y = "Mean Hub Height (m)")

ggplot(aggregated_data, aes(x = Operating_Year, y = Mean_Rotor_Diameter)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Evolution of Mean Rotor Diameter Over the Years", 
       x = "Year", y = "Mean Rotor Diameter (m)")

# Determine top states by number of turbines and by capacity
top_turbines_by_state <- data %>% count(t_state) %>% arrange(desc(n)) %>% head(5)
top_capacity_by_state <- aggregated_data %>%
  group_by(State) %>%
  summarise(Total_Capacity_MW = sum(Mean_Project_Capacity_MW, na.rm = TRUE)) %>%
  arrange(desc(Total_Capacity_MW)) %>%
  head(5)

# Calculate the total installed wind capacity in the US
total_us_capacity <- sum(aggregated_data$Mean_Project_Capacity_MW, na.rm = TRUE)

# Create tables for display
kable(list("Top Five States by Number of Turbines" = top_turbines_by_state,
           "Top Five States by Capacity" = top_capacity_by_state,
           "Total US Capacity" = total_us_capacity))
