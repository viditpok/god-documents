# Set working directory
setwd("/Users/viditpokharna/Downloads")

# Part 1: Preparing data for the analysis

# (a) Download the data into R studio
airbnb_data <- read.csv("airbnb_data.csv")

# (b) Remove observations with missing or NA values in price
airbnb_data <- na.omit(airbnb_data[!is.na(airbnb_data$price), ])

# (c) Remove observations with NA values in specified columns
cols_to_check <- c("accommodates", "beds", "number_of_reviews", "review_scores_rating")
airbnb_data <- airbnb_data[complete.cases(airbnb_data[, cols_to_check]), ]

# (d) Create a variable called host_experience
# Convert host_since from a factor to a Date object first if not already
airbnb_data$host_since <- as.Date(airbnb_data$host_since)
# Calculate host_experience in years
airbnb_data$host_experience <- (as.Date("2023-06-05") - airbnb_data$host_since) / 365
# Remove observations with missing host_experience
airbnb_data <- airbnb_data[!is.na(airbnb_data$host_experience), ]

# (e) Create a variable called entire_apt
airbnb_data$entire_apt <- ifelse(airbnb_data$room_type == "Entire home/apt", 1, 0)
# Assuming you want to remove NA values after this transformation
airbnb_data <- airbnb_data[!is.na(airbnb_data$entire_apt), ]

# (f) Create a variable host_is_superhost
airbnb_data$host_response_rate <- as.numeric(gsub("%", "", airbnb_data$host_response_rate)) / 100
airbnb_data$host_is_superhost <- ifelse(airbnb_data$host_response_rate >= 0.90 & airbnb_data$number_of_reviews >= 10 & airbnb_data$review_scores_rating >= 4.8, 1, 0)
# Remove observations with NA values in the new variable
airbnb_data <- airbnb_data[!is.na(airbnb_data$host_is_superhost), ]

# (g) Sort your data by id
airbnb_data <- airbnb_data[order(airbnb_data$id), ]

# Part 2: Analysis

# (a) Set the seed to ensure reproducible results
set.seed(0)

# (b) Split data into training and test samples
sample_size <- nrow(airbnb_data)
test_index <- sample(1:sample_size, size = sample_size / 2)
test_data <- airbnb_data[test_index, ]
train_data <- airbnb_data[-test_index, ]

# (c) Estimate a linear regression model and compute R squared and MSE for the test sample
model_linear <- lm(price ~ accommodates + beds + host_experience + host_is_superhost + entire_apt + number_of_reviews + review_scores_rating, data = train_data)
summary(model_linear) # To get the R squared value and coefficients

# Calculate MSE for the test sample
predictions_linear <- predict(model_linear, test_data)
mse_linear <- mean((test_data$price - predictions_linear)^2)
rsquared_linear <- summary(model_linear)$r.squared

# (d) Estimate a linear model with second order polynomials and interactions
model_poly <- lm(price ~ poly(accommodates, 2) + poly(beds, 2) + poly(host_experience, 2) + poly(number_of_reviews, 2) + poly(review_scores_rating, 2) +
                   accommodates:beds + accommodates:host_experience + accommodates:number_of_reviews + accommodates:review_scores_rating +
                   beds:host_experience + beds:number_of_reviews + beds:review_scores_rating +
                   host_experience:number_of_reviews + host_experience:review_scores_rating +
                   number_of_reviews:review_scores_rating +
                   host_is_superhost + entire_apt,
                 data = train_data)
# Calculate MSE and R squared for the test sample
predictions_poly <- predict(model_poly, test_data)
mse_poly <- mean((test_data$price - predictions_poly)^2)
rsquared_poly <- summary(model_poly)$r.squared

# (e) Perform backward stepwise selection
library(MASS) # for stepAIC
model_step <- stepAIC(model_poly, direction = "backward", k = log(nrow(train_data))) # BIC adjustment
summary(model_step) # To get the final model after selection

# Calculate MSE for the test sample
predictions_step <- predict(model_step, test_data)
mse_step <- mean((test_data$price - predictions_step)^2)

# (f) Fit Ridge and Lasso regression models
library(glmnet)
x <- model.matrix(price ~ accommodates + beds + host_experience + host_is_superhost + entire_apt + number_of_reviews + review_scores_rating - 1, data = train_data)
y <- train_data$price

# Set alpha = 0 for Ridge and alpha = 1 for Lasso
ridge_model <- glmnet(x, y, alpha = 0, lambda = c(0, 5, 10))
lasso_model <- glmnet(x, y, alpha = 1, lambda = c(0, 5, 10))

# Compute MSE for the test sample for each model
x_test <- model.matrix(price ~ accommodates + beds + host_experience + host_is_superhost + entire_apt + number_of_reviews + review_scores_rating - 1, data = test_data)
y_test <- test_data$price

predictions_ridge <- predict(ridge_model, s = c(0, 5, 10), newx = x_test)
mse_ridge <- colMeans((y_test - predictions_ridge)^2)

predictions_lasso <- predict(lasso_model, s = c(0, 5, 10), newx = x_test)
mse_lasso <- colMeans((y_test - predictions_lasso)^2)

# (g) Perform 10-fold cross-validation for Lasso and Ridge
set.seed(0) # for reproducibility
cv_ridge <- cv.glmnet(x, y, alpha = 0)
cv_lasso <- cv.glmnet(x, y, alpha = 1)

# Compute MSE for the best models
best_lambda_ridge <- cv_ridge$lambda.min
best_lambda_lasso <- cv_lasso$lambda.min

predictions_cv_ridge <- predict(ridge_model, s = best_lambda_ridge, newx = x_test)
mse_cv_ridge <- mean((y_test - predictions_cv_ridge)^2)

predictions_cv_lasso <- predict(lasso_model, s = best_lambda_lasso, newx = x_test)
mse_cv_lasso <- mean((y_test - predictions_cv_lasso)^2)

# Part 3: Repeating analysis with variations

run_analysis <- function(add_noise = FALSE, train_size = 0.5) {
  # Set the seed
  set.seed(0)
  
  # Add noise variables if specified
  if (add_noise) {
    airbnb_data$noise1 <- airbnb_data$host_experience + rnorm(nrow(airbnb_data), mean = 0, sd = 0.01)
    airbnb_data$noise2 <- airbnb_data$host_is_superhost + rnorm(nrow(airbnb_data), mean = 0, sd = 0.01)
    airbnb_data$noise3 <- airbnb_data$number_of_reviews + rnorm(nrow(airbnb_data), mean = 0, sd = 0.01)
  }
  
  # Splitting data into training and test sets based on the specified training size
  sample_size <- floor(train_size * nrow(airbnb_data))
  train_indices <- sample(seq_len(nrow(airbnb_data)), size = sample_size)
  train_data <- airbnb_data[train_indices, ]
  test_data <- airbnb_data[-train_indices, ]
  
  # Define the model formula
  base_covariates <- c("accommodates", "beds", "host_experience", "host_is_superhost", "entire_apt", "number_of_reviews", "review_scores_rating")
  if (add_noise) {
    base_covariates <- c(base_covariates, "noise1", "noise2", "noise3")
  }
  formula <- as.formula(paste("price ~ ", paste(base_covariates, collapse = " + ")))
  
  # Estimate the linear regression model
  model <- lm(formula, data = train_data)
  
  # Compute R squared and MSE on the test sample
  test_predictions <- predict(model, newdata = test_data)
  r_squared <- summary(model)$r.squared
  mse <- mean((test_predictions - test_data$price)^2)
  
  # Return results
  list(model = model, R_squared = r_squared, MSE = mse)
}

results_50 <- run_analysis(FALSE, 0.5)  # 50% training data, no noise
results_10 <- run_analysis(FALSE, 0.1)  # 10% training data, no noise
results_2 <- run_analysis(FALSE, 0.02)  # 2% training data, no noise
results_50_noise <- run_analysis(TRUE, 0.5)  # 50% training data with noise
results_10_noise <- run_analysis(TRUE, 0.1)  # 10% training data with noise
results_2_noise <- run_analysis(TRUE, 0.02)  # 2% training data with noise