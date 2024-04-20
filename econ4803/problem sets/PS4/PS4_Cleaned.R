# Assignment: PS4
# Author: Vidit Pokharna
# Due date: 04/20/2024
# Group members: Matthew, Tannishtha, Roshani

# Clear the environment
rm(list=ls())

### Problem 1
###############################################
## Part 1a: Load necessary libraries
library(neuralnet)
library(randomForest)
library(keras)
library(doParallel)
library(tensorflow)
library(dplyr)
library(magrittr)
## Part 1b: Draw observable explanatory variables
n <- 1000
x1 <- rgamma(n, 2, 1)
x2 <- rnorm(n, 0, 2)
x3 <- rweibull(n, 2, 2)
x4 <- rlogis(n, 2, 1)
x5 <- rbeta(n, 2, 1)
x <- cbind(x1, x2, x3, x4, x5)
# Transform into independent random variables by de-correlating
c1 <- var(x)
chol1 <- solve(chol(c1))
x <- x %*% chol1
# Generate and apply a random correlation matrix to introduce new dependencies
R <- matrix(runif(ncol(x)^2, -1, 1), ncol = ncol(x))
RtR <- R %*% t(R)
corr <- cov2cor(RtR)
x <- x %*% chol(corr)
datam <- as.data.frame(x)
colnames(datam) <- c("x1", "x2", "x3", "x4", "x5")
## Part 1c: Model Data Preparation
set.seed(0)
# Generate targets based on two different specifications
datam$y1 <- datam$x1 + (datam$x3 * datam$x2^2) / 10 + (datam$x4 * datam$x1 * datam$x5) / 10
datam$y2 <- log(abs(datam$x1^4 / 10) + abs(datam$x2) + datam$x3^2) + (datam$x4 * datam$x2 * sin(datam$x5)) + rnorm(nrow(datam))
# Split the data into training and testing sets
datam_split <- sample(seq_len(nrow(datam)), size = 0.5 * nrow(datam))
data_train <- datam[-datam_split, ]
data_test <- datam[datam_split, ]
# Neural Network Modeling
# Define the architecture of the neural network for both specifications
hidden_layers <- c(64, 32, 16)  # Hidden layers configuration
neural1_1c <- neuralnet(y1 ~ x1 + x2 + x3 + x4 + x5, data = data_train, hidden = hidden_layers)
neural2_1c <- neuralnet(y2 ~ x1 + x2 + x3 + x4 + x5, data = data_train, hidden = hidden_layers)
# Evaluate the models on test data
pred1c_neural1 <- predict(neural1_1c, data_test)
neural1_mse <- mean((pred1c_neural1 - data_test$y1)^2)
pred1c_neural2 <- predict(neural2_1c, data_test)
neural2_mse <- mean((pred1c_neural2 - data_test$y2)^2)
# Statistical Modeling using Polynomial Regression
series1c_1 <- lm(y1 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3), data = data_train)
series1c_2 <- lm(y2 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3), data = data_train)
pred1c_series1 <- predict(series1c_1, data_test)
series1_mse <- mean((pred1c_series1 - data_test$y1)^2)
pred1c_series2 <- predict(series1c_2, data_test)
series2_mse <- mean((pred1c_series2 - data_test$y2)^2)
# Random Forest Modeling
forest1c_1 <- randomForest(y1 ~ x1 + x2 + x3 + x4 + x5, data = data_train, ntree = 1000, mtry = 4)
forest1c_2 <- randomForest(y2 ~ x1 + x2 + x3 + x4 + x5, data = data_train, ntree = 1000, mtry = 4)
pred1c_forest1 <- predict(forest1c_1, data_test)
forest1_mse <- mean((pred1c_forest1 - data_test$y1)^2)
pred1c_forest2 <- predict(forest1c_2, data_test)
forest2_mse <- mean((pred1c_forest2 - data_test$y2)^2)
## Part 1d: Parallel Computing for Model Evaluations
# Set up parallel computing to speed up the simulations
nsim <- 50  # Number of simulations
nCores <- detectCores()
cl <- makeCluster(nCores)
registerDoParallel(cl)
# Run simulations in parallel to estimate MSE across different models and specifications
results <- foreach(i = 1:nsim, .combine = rbind, .packages = c('neuralnet', 'randomForest')) %dopar% {
  # Sample data for training and testing
  datam_split <- sample(seq_len(nrow(datam)), size = 0.5 * nrow(datam))
  data_train <- datam[-datam_split, ]
  data_test <- datam[datam_split, ]
  # Define and train models
  neural_1 <- neuralnet(y1 ~ x1 + x2 + x3 + x4 + x5, data = data_train, hidden = c(64, 32, 16))
  neural_2 <- neuralnet(y2 ~ x1 + x2 + x3 + x4 + x5, data = data_train, hidden = c(64, 32, 16))
  pred_neural1 <- predict(neural_1, data_test)
  pred_neural2 <- predict(neural_2, data_test)
  neural1_mse <- mean((pred_neural1 - data_test$y1)^2)
  neural2_mse <- mean((pred_neural2 - data_test$y2)^2)
  series_1 <- lm(y1 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3), data = data_train)
  series_2 <- lm(y2 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3), data = data_train)
  pred_series1 <- predict(series_1, data_test)
  pred_series2 <- predict(series_2, data_test)
  series1_mse <- mean((pred_series1 - data_test$y1)^2)
  series2_mse <- mean((pred_series2 - data_test$y2)^2)
  forest_1 <- randomForest(y1 ~ x1 + x2 + x3 + x4 + x5, data = data_train, ntree = 1000, mtry = 4)
  forest_2 <- randomForest(y2 ~ x1 + x2 + x3 + x4 + x5, data = data_train, ntree = 1000, mtry = 4)
  pred_forest1 <- predict(forest_1, data_test)
  pred_forest2 <- predict(forest_2, data_test)
  forest1_mse <- mean((pred_forest1 - data_test$y1)^2)
  forest2_mse <- mean((pred_forest2 - data_test$y2)^2)
  # Return all MSEs for this iteration
  c(neural1_mse, neural2_mse, series1_mse, series2_mse, forest1_mse, forest2_mse)
}
# Clean up resources used by parallel processing
stopCluster(cl)
rm(cl)
# Report average MSEs from simulations
neural_sp1_mse <- mean(results[, 1])
neural_sp2_mse <- mean(results[, 2])
series_sp1_mse <- mean(results[, 3])
series_sp2_mse <- mean(results[, 4])
forest_sp1_mse <- mean(results[, 5])
forest_sp2_mse <- mean(results[, 6])

#-------------------------------------------------------------------------------------------#

# Clear the environment
rm(list=ls())

### Problem 2
###############################################
# Load necessary libraries
library(dplyr)
library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(stats)
library(knitr)
library(Metrics)
# Load data and clean initial variables
airbnb_data <- read.csv("./Downloads/airbnb_data.csv")
airbnb_data$X <- NULL
# Data Cleaning and Preprocessing
# Select complete cases for specified columns, and compute host experience
airbnb <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating")]),]
airbnb$host_experience <- as.double(difftime(as.Date("2023-06-01"), as.Date(airbnb$host_since), units = "days")) / 365
airbnb <- airbnb[complete.cases(airbnb[, "host_experience"]),]
airbnb$entire_apt <- ifelse(airbnb$room_type == "Entire home/apt", 1, 0)
airbnb$host_is_superhost <- ifelse((airbnb$host_response_rate >= 90 & airbnb$number_of_reviews >= 10 & airbnb$review_scores_rating >= 4.8), 1, 0)
airbnb <- airbnb[complete.cases(airbnb[, "host_is_superhost"]),]
airbnb <- airbnb[order(airbnb$id),]
# Splitting data into training and test sets
train_indices <- sample(1:nrow(airbnb), size = ceiling(nrow(airbnb) * 0.9), replace = FALSE)
test_indices <- setdiff(1:nrow(airbnb), train_indices)
# Model matrix preparation for regression
model.mat <- model.matrix(price ~ (accommodates + beds + host_experience + number_of_reviews + review_scores_rating)^2 + host_is_superhost + entire_apt + 
                            I(accommodates^2) + I(beds^2) + I(host_experience^2) + I(number_of_reviews^2) + I(review_scores_rating^2), data = airbnb)
model_df <- as.data.frame(model.mat)
model_df$`(Intercept)` <- NULL
model_df$price <- airbnb$price
## Part 2a: Principal Component Analysis
pca_model <- prcomp(model_df[, -which(names(model_df) == "price")], center = TRUE, scale. = TRUE)
pve <- summary(pca_model)$importance[2, 1:4]  # Proportion of Variance Explained
cum_pve <- cumsum(pve)  # Cumulative Proportion of Variance Explained
cat("Proportion of Variance Explained by each of the first 4 PCs:", pve, "\n")
cat("Cumulative Proportion of Variance Explained by the first 4 PCs:", cum_pve, "\n")
## Part 2b: Regression Analysis using Principal Components
top4_pc <- pca_model$x[, 1:4]  # Extract top 4 principal components
model_df$PC1 <- top4_pc[, 1]
model_df$PC2 <- top4_pc[, 2]
model_df$PC3 <- top4_pc[, 3]
model_df$PC4 <- top4_pc[, 4]
airbnb_train <- model_df[train_indices, ]
airbnb_test <- model_df[test_indices, ]
price <- airbnb$price[train_indices]
# Regression model using top 4 PCs
pc_model <- lm(price ~ PC1 + PC2 + PC3 + PC4, data = airbnb_train)
predictions_pc <- predict(pc_model, newdata = airbnb_test)
mse_pc <- mse(airbnb$price[test_indices], predictions_pc)
# Polynomial regression for comparison
poly_model <- lm(price ~ . -PC1 -PC2 -PC3 -PC4, data = airbnb_train)
predictions_poly <- predict(poly_model, newdata = airbnb_test)
mse_poly <- mse(airbnb$price[test_indices], predictions_poly)
# Lasso and Ridge regression for regularization
lasso.reg <- glmnet(model.mat[train_indices, -which(names(model.mat) == "price")], price[train_indices], alpha = 1)
lasso.cv <- cv.glmnet(model.mat[train_indices, -which(names(model.mat) == "price")], price[train_indices], alpha = 1)
lambda_hat <- lasso.cv$lambda.min
lasso.cv.pred <- predict(lasso.reg, s = lambda_hat, newx = model.mat[test_indices, -which(names(model.mat) == "price")])
mse_lasso <- mse(airbnb$price[test_indices], lasso.cv.pred)
ridge.reg <- glmnet(model.mat[train_indices, -which(names(model.mat) == "price")], price[train_indices], alpha = 0)
ridge.cv <- cv.glmnet(model.mat[train_indices, -which(names(model.mat) == "price")], price[train_indices], alpha = 0)
lambda_hat <- ridge.cv$lambda.min
ridge.cv.pred <- predict(ridge.reg, s = lambda_hat, newx = model.mat[test_indices, -which(names(model.mat) == "price")])
mse_ridge <- mse(airbnb$price[test_indices], ridge.cv.pred)
# Compile results into a dataframe and display
mse_results2 <- data.frame(PCA = mse_pc, Polynomial = mse_poly, Ridge = mse_ridge, Lasso = mse_lasso)
print(mse_results2)

#-------------------------------------------------------------------------------------------#

# Clear the environment
rm(list=ls())

### Problem 3
###############################################
# Load necessary libraries
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(stats)
# Load the Airbnb dataset
airbnb_data <- read.csv("./Downloads/airbnb_data.csv")
airbnb_data$X <- NULL
# Select complete cases and calculate host experience in years
datam <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating", "host_since")]),]
datam$host_experience <- as.double(difftime(as.Date("2023-06-01"), as.Date(datam$host_since), units = "days")) / 365
datam <- datam[complete.cases(datam[, "host_experience"]),]
# Encode categorical data and ensure all necessary data is complete
datam$entire_apt <- ifelse(datam$room_type == "Entire home/apt", 1, 0)
datam$host_is_superhost <- ifelse(datam$host_response_rate >= 90 & datam$number_of_reviews >= 10 & datam$review_scores_rating >= 4.8, 1, 0)
datam <- datam[complete.cases(datam[, "host_is_superhost"]),]
datam <- datam[order(datam$id),]  # Order data by ID
# Additional data cleaning steps
datam$host_identity_verified <- ifelse(datam$host_identity_verified == "t", 1, 0)
datam <- datam[complete.cases(datam[, c("review_scores_rating", "review_scores_accuracy", "review_scores_value")]),]
# Split data into training and testing sets
set.seed(0)
train_indices <- sample(1:nrow(datam), 0.9 * nrow(datam))
test_indices <- setdiff(1:nrow(datam), train_indices)
data_train <- datam[train_indices, ]
y_test <- datam$host_is_superhost[test_indices]
# Cross-validate a logistic regression model using lasso regularization
x <- model.matrix(host_is_superhost ~ .^2, data = datam)  # Polynomial terms for interaction
lasso.logit <- cv.glmnet(x[train_indices,], datam$host_is_superhost[train_indices], alpha = 1, family = "binomial")
lambda_optimal <- lasso.logit$lambda.min  # Optimal lambda from cross-validation
# Predict and calculate mean classification error (MCE) for logistic regression
predictions_lasso <- predict(lasso.logit, s = lambda_optimal, newx = x[test_indices,], type = "response")
mce_lasso <- mean((predictions_lasso >= 0.5) != y_test)
## Part 3a: Random Forest on Predictors
# Train a random forest classifier
forest_model <- randomForest(host_is_superhost ~ .^2, data = data_train, ntree = 1000, mtry = 4)
predictions_forest <- predict(forest_model, datam[test_indices,])
mce_forest <- mean(predictions_forest != y_test)
# MCE comparison output
mce_comparison <- data.frame(Lasso = mce_lasso, Random_Forest = mce_forest)
print(mce_comparison)
## Part 3b: K-means Clustering and SVM
# Clustering setup
scaled_data <- scale(data_train[-which(names(data_train) == "host_is_superhost")])  # Exclude target variable
kmeans_result <- kmeans(scaled_data, centers = 1000)
data_train$cluster <- kmeans_result$cluster
# Summarize clustered data
summary_data <- aggregate(. ~ cluster, data = data_train, FUN = mean)
summary_data$host_is_superhost <- ifelse(summary_data$host_is_superhost >= 0.5, 1, 0)  # Binarize average outcome
# Train SVM on summarized data
svm_fit <- svm(host_is_superhost ~ ., data = summary_data, kernel = "radial", cost = 10, gamma = 0.01)
predictions_svm <- predict(svm_fit, summary_data)
mce_svm <- mean(predictions_svm != summary_data$host_is_superhost)
# Output results
print(paste("Mean Classification Error for SVM on Clustered Data:", mce_svm))