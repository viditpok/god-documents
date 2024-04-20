# Set the working directory and load the dplyr package
setwd("/Users/viditpokharna/Downloads")
library(dplyr)

# Read the wind resource data and perform initial cleaning
wind_resource_data <- read.csv("wtk_site_metadata.csv")
wind_resource_data <- subset(wind_resource_data,
                             !(is.na(State) | is.na(County) | power_curve == "offshore"))
wind_resource_data <- subset(wind_resource_data, 
                             !(State %in% c("AK", "HI")))

# Calculate summary statistics for wind resource data
wind_summary <- wind_resource_data %>%
  summarise(
    Mean_Wind_Speed = mean(wind_speed, na.rm = TRUE),
    SD_Wind_Speed = sd(wind_speed, na.rm = TRUE),
    Min_Wind_Speed = min(wind_speed, na.rm = TRUE),
    Max_Wind_Speed = max(wind_speed, na.rm = TRUE),
    Mean_Capacity_Factor = mean(capacity_factor, na.rm = TRUE),
    SD_Capacity_Factor = sd(capacity_factor, na.rm = TRUE),
    Min_Capacity_Factor = min(capacity_factor, na.rm = TRUE),
    Max_Capacity_Factor = max(capacity_factor, na.rm = TRUE),
    Mean_Usable_Area = mean(fraction_of_usable_area, na.rm = TRUE),
    SD_Usable_Area = sd(fraction_of_usable_area, na.rm = TRUE),
    Min_Usable_Area = min(fraction_of_usable_area, na.rm = TRUE),
    Max_Usable_Area = max(fraction_of_usable_area, na.rm = TRUE)
  )

# Identify the top five states with the highest mean wind speed
top_wind_speed_states <- wind_resource_data %>%
  group_by(State) %>%
  summarise(Mean_Wind_Speed = mean(wind_speed, na.rm = TRUE)) %>%
  arrange(desc(Mean_Wind_Speed)) %>%
  head(5)

# Identify the top five states with the highest mean capacity factor
top_capacity_factor_states <- wind_resource_data %>%
  group_by(State) %>%
  summarise(Mean_Capacity_Factor = mean(capacity_factor, na.rm = TRUE)) %>%
  arrange(desc(Mean_Capacity_Factor)) %>%
  head(5)
