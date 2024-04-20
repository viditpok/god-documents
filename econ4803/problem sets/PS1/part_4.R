# Set the working directory and load required libraries
setwd("/Users/viditpokharna/Downloads")
library(dplyr)
library(ggplot2)
library(sf)

# Load the datasets
wind_turbines <- read.csv("uswtdb_v6_1_20231128.csv")
wind_ordinances <- read.csv("wind_ordinance_main.csv")

state_abbreviations <- c(AL = "Alabama", AK = "Alaska", AZ = "Arizona", AR = "Arkansas", CA = "California", 
                         CO = "Colorado", CT = "Connecticut", DE = "Delaware", FL = "Florida", GA = "Georgia", 
                         HI = "Hawaii", ID = "Idaho", IL = "Illinois", IN = "Indiana", IA = "Iowa", 
                         KS = "Kansas", KY = "Kentucky", LA = "Louisiana", ME = "Maine", MD = "Maryland", 
                         MA = "Massachusetts", MI = "Michigan", MN = "Minnesota", MS = "Mississippi", MO = "Missouri", 
                         MT = "Montana", NE = "Nebraska", NV = "Nevada", NH = "New Hampshire", NJ = "New Jersey", 
                         NM = "New Mexico", NY = "New York", NC = "North Carolina", ND = "North Dakota", OH = "Ohio", 
                         OK = "Oklahoma", OR = "Oregon", PA = "Pennsylvania", RI = "Rhode Island", SC = "South Carolina", 
                         SD = "South Dakota", TN = "Tennessee", TX = "Texas", UT = "Utah", VT = "Vermont", 
                         VA = "Virginia", WA = "Washington", WV = "West Virginia", WI = "Wisconsin", WY = "Wyoming")

# Convert state abbreviations to full names
wind_turbines$t_state <- sapply(wind_turbines$t_state, function(abb) {
  if (!is.na(abb) && abb %in% names(state_abbreviations)) {
    return(state_abbreviations[[abb]])
  }
  return(abb)
}, USE.NAMES = FALSE)

# Clean county names
wind_turbines$t_county <- gsub(" County", "", wind_turbines$t_county)

# Combine unique counties from both datasets
all_counties <- unique(rbind(
  data.frame(County = unique(wind_turbines$t_county)),
  data.frame(County = unique(wind_ordinances$County))
))

# Merge to create a master list of counties with ordinance information
counties_master <- merge(all_counties, wind_ordinances, by = "County", all.x = TRUE)
counties_master$has_ordinance <- ifelse(is.na(counties_master$ordinance_year), 0, 1)

# Merge turbines with the master list of counties
merged_data <- merge(wind_turbines, counties_master, by.x = "t_county", by.y = "County")

# Summarize data at the county level
county_level_data <- merged_data %>%
  group_by(t_state, t_county, p_year) %>%
  summarize(
    total_capacity = sum(t_cap, na.rm = TRUE),
    average_rotor_diameter = mean(t_rd, na.rm = TRUE),
    has_ordinance = max(has_ordinance, na.rm = TRUE)
  )

# Filter for years 2001 to 2022 and calculate average capacity
county_level_data_avg <- county_level_data %>%
  filter(p_year >= 2001, p_year <= 2022) %>%
  group_by(t_state, t_county, p_year, has_ordinance) %>%
  summarize(average_capacity = mean(total_capacity, na.rm = TRUE)) %>%
  ungroup()

# Create a scatter plot of average wind capacity
ggplot(county_level_data_avg, aes(x = p_year, y = average_capacity, color = as.factor(has_ordinance))) +
  geom_point() +
  labs(
    title = "Average Wind Capacity of Counties With & Without Wind Ordinance (2001-2022)",
    x = "Year",
    y = "Average Wind Capacity (MW)",
    color = "Wind Ordinance"
  ) +
  scale_color_manual(values = c("red", "blue"), labels = c("Without Ordinance", "With Ordinance")) +
  theme_minimal()

# Summarize data at the project level
project_level_data <- merged_data %>%
  group_by(p_name) %>%
  summarize(
    total_capacity = sum(t_cap, na.rm = TRUE),
    average_rotor_diameter = mean(t_rd, na.rm = TRUE),
    has_ordinance = max(has_ordinance, na.rm = TRUE)
  )

# Create a summary table
summary_table <- project_level_data %>%
  group_by(has_ordinance) %>%
  summarize(
    average_capacity = mean(total_capacity, na.rm = TRUE),
    average_rotor_diameter = mean(average_rotor_diameter, na.rm = TRUE)
  )