# Assignment: PS4
# Author: Vidit Pokharna
# Due date: 04/20/2024
# Group members: Matthew, Tannishtha, Roshani

# Clear the environment
rm(list=ls())

## 1. Empirical Problems
###############################################
# 1b. Draw observable explanatory variables
n <- 1000
x1 = rgamma(n,2,1)
x2 = rnorm(n,0,2)
x3 = rweibull(n,2,2)
x4 = rlogis(n,2,1)
x5 = rbeta(n,2,1)
x = cbind(x1,x2,x3,x4,x5)
###############################################
#transform into independent random variables
# find the current correlation matrix
c1 <- var(x)
# cholesky decomposition to get independence
chol1 <- solve(chol(c1))
x <- x %*% chol1
###############################################
#generate random correlation matrix
R <- matrix(runif(ncol(x)^2,-1,1), ncol=ncol(x))
RtR <- R %*% t(R)
corr <- cov2cor(RtR)
# check that it is positive definite
sum((eigen(corr)$values>0))==ncol(x)
################################################
#transform according to this correlation matrix
x <- x %*% chol(corr)
datam <- as.data.frame(x)
datam <- datam %>% dplyr::rename(x1 = V1, x2 = V2, x3 = V3, x4 = V4, x5 = V5)





## 1(c)

#Set see to 0
set.seed(0)

#----------Data with Specification 1-------#
y1 = datam$x1 + (datam$x3 * datam$x2^2)/10 + (datam$x4 * datam$x1 * datam$x5)/10
y2 = log(abs(datam$x1^4/10) + abs(datam$x2) + datam$x3^2) + 
  (datam$x4 * datam$x2 * sin(datam$x5)) + rnorm(nrow(datam))
datam$y1 = y1
datam$y2 = y2

#  Randomly allocating 50% of the data to the test sample and the rest to the training sample.
datam_split = sample(seq_len(nrow(datam)), nrow(datam)*.5)

# Training and test sample specification 1
data_train <- datam[-datam_split,]
data_test <- datam[datam_split,]

# Create 3 hidden layers first layer with 64, second layer with 32, and the third layer with 16 neurons respectively
hidden_layers <- c(64, 32, 16)

#Estimate the neural net
neural1_1c = neuralnet(y1 ~ x1 + x2 + x3 + x4 + x5,
                       data = data_train,
                       hidden=hidden_layers)

neural2_1c = neuralnet(y2 ~ x1 + x2 + x3 + x4 + x5,
                       data = data_train,
                       hidden=hidden_layers)

# To view our model architecture
plot(neural1_1c,rep = "best")
plot(neural2_1c,rep = "best")

# Estimate MSE for neural net 
pred1c_neural1 <- predict(neural1_1c, data_test)
neural1_mse = mean((pred1c_neural1 - data_test$y1)^2)

pred1c_neural2 <- predict(neural2_1c, data_test)
neural2_mse = mean((pred1c_neural2 - data_test$y2)^2)

#-----------------------------#
## Estimate a series using the poly function. Set the degree to 3.

series1c_1 = lm(y1 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3),
                data = data_train)
series1c_2 = lm(y2 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3),
                data = data_train)

# Estimate MSE for series using the poly function 
pred1c_series1 <- predict(series1c_1, data_test)
series1_mse = mean((pred1c_series1 - data_test$y1)^2)

pred1c_series2 <- predict(series1c_2, data_test)
series2_mse = mean((pred1c_series2 - data_test$y2)^2)

#-----------------------------#
## Estimate a random forest. Use 1000 trees with 4 covariates sampled time.
forest1c_1 = randomForest(y1 ~ x1 + x2 + x3 + x4 + x5,
                          data = data_train,
                          ntree = 1000,
                          mtry = 4)
forest1c_2 = randomForest(y2 ~ x1 + x2 + x3 + x4 + x5,
                          data = data_train,
                          ntree = 1000,
                          mtry = 4)
# Estimate MSE for neural net 
pred1c_forest1 <- predict(forest1c_1, data_test)
forest1_mse = mean((pred1c_forest1 - data_test$y1)^2)

pred1c_forest2 <- predict(forest1c_2, data_test)
forest2_mse = mean((pred1c_forest2 - data_test$y2)^2)



## Question 1d
# parallel computing setup
nsim = 50
nCores = parallel::detectCores()
cl = parallel::makeCluster(nCores)
doParallel::registerDoParallel(cl)

results = foreach(i=1:nsim, .combine=rbind, .packages = c('neuralnet', 'randomForest')) %dopar% {
  #  Randomly allocating 50% of the data to the test sample and the rest to the training sample.
  datam_split = sample(seq_len(nrow(datam)), nrow(datam)*.5)
  
  # Training and test sample specification 1
  data_train <- datam[-datam_split,]
  data_test <- datam[datam_split,]
  
  
  # estimate a neural net with 3 hidden layers (64, 32, and 16 neurons)
  neural_1 = neuralnet(y1 ~ x1 + x2 + x3 + x4 + x5,
                       data = data_train,
                       hidden = c(64, 32, 16))
  neural_2 = neuralnet(y2 ~ x1 + x2 + x3 + x4 + x5,
                       data = data_train,
                       hidden = c(64, 32, 16),
                       threshold = 0.1)
  
  # Estimate MSE for neural net 
  pred_neural1 <- predict(neural_1, data_test)
  neural1_mse = mean((pred_neural1 - data_test$y1)^2)
  
  pred_neural2 <- predict(neural_2, data_test)
  neural2_mse = mean((pred_neural2 - data_test$y2)^2)
  
  #-------#
  # estimate a series using the poly function
  series_1 = lm(y1 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3),
                data = data_train)
  series_2 = lm(y2 ~ poly(x1, 3) + poly(x2, 3) + poly(x3, 3) + poly(x4, 3) + poly(x5, 3),
                data = data_train)
  
  # Estimate MSE for series using the poly function 
  pred_series1 <- predict(series_1, data_test)
  series1_mse = mean((pred_series1 - data_test$y1)^2)
  
  pred_series2 <- predict(series_2, data_test)
  series2_mse = mean((pred_series2 - data_test$y2)^2)
  
  #-----#
  
  # estimate a random forest
  forest_1 = randomForest(y1 ~ x1 + x2 + x3 + x4 + x5,
                          data = data_train,
                          ntree = 1000,
                          mtry = 4)
  forest_2 = randomForest(y2 ~ x1 + x2 + x3 + x4 + x5,
                          data = data_train,
                          ntree = 1000,
                          mtry = 4)
  
  # Estimate MSE for neural net 
  pred_forest1 <- predict(forest_1, data_test)
  forest1_mse = mean((pred_forest1 - data_test$y1)^2)
  
  pred_forest2 <- predict(forest_2, data_test)
  forest2_mse = mean((pred_forest2 - data_test$y2)^2)
  
  c(neural1_mse, neural2_mse, series1_mse, series2_mse, forest1_mse,
    forest2_mse)
}

#cleanUp
parallel::stopCluster(cl)
rm(cl)
# report results
neural_sp1_mse <- print(c(mean(results[,1])))
neural_sp2_mse <- print(c(mean(results[,2])))
series_sp1_mse <- print(c(mean(results[,3])))
series_sp2_mse <- print(c(mean(results[,4])))
forest_sp1_mse <- print(c(mean(results[,5])))
forest_sp2_mse <- print(c(mean(results[,6])))

## 1(e)
# Prepare data - Ensure train and test sets have the same number of feature columns
train_x <- as.matrix(data_train[, c("x1", "x2", "x3", "x4", "x5")])  # Independent variables
train_y1 <- data_train$y1  # Target variable for Specification 1
train_y2 <- data_train$y2  # Target variable for Specification 2

test_x <- as.matrix(data_test[, c("x1", "x2", "x3", "x4", "x5")])  # Independent variables for testing
test_y1 <- data_test$y1  # Test target for Specification 1
test_y2 <- data_test$y2  # Test target for Specification 2

# Confirm the number of features
num_features <- ncol(train_x)
print(paste("Number of features:", num_features))

## Build and Train Neural Network Models
# Model for Specification 1
model_spec1 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(num_features)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "linear")

# Use RMSprop optimizer
optimizer_rms <- optimizer_rmsprop(lr = 0.001)

model_spec1 %>% compile(
  optimizer = optimizer_rms,
  loss = 'mse',
  metrics = c('mean_absolute_error')
)

# Fit model for Specification 1
model_spec1 %>% fit(
  train_x, train_y1,
  epochs = 20,
  batch_size = 16,
  validation_data = list(test_x, test_y1)
)

# Model for Specification 2
model_spec2 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(num_features)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "linear")

model_spec2 %>% compile(
  optimizer = optimizer_rms,
  loss = 'mse',
  metrics = c('mean_absolute_error')
)

# Fit model for Specification 2
model_spec2 %>% fit(
  train_x, train_y2,
  epochs = 20,
  batch_size = 16,
  validation_data = list(test_x, test_y2)
)

## Evaluate Models
# Test MSE for Specification 1
score_spec1 <- model_spec1 %>% evaluate(test_x, test_y1, verbose = 0)
print(paste("Test MSE for Specification 1:", score_spec1[[1]]))
print(paste("Mean Absolute Error for Specification 1:", score_spec1[[2]]))

# Test MSE for Specification 2
score_spec2 <- model_spec2 %>% evaluate(test_x, test_y2, verbose = 0)
print(paste("Test MSE for Specification 2:", score_spec2[[1]]))
print(paste("Mean Absolute Error for Specification 2:", score_spec2[[2]]))



#-------------------------------------------------------------------------------------------#

# Clear the environment
rm(list=ls())

# Load necessary libraries
library(dplyr)
library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(stats)
library(knitr)
library(Metrics)

# ---------------------------------
# Question 2
# ---------------------------------

# Load and prepare the data
airbnb_data <- read.csv("./Downloads/airbnb_data.csv")
airbnb_data$X <- NULL

airbnb <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating")]),]
airbnb$host_experience <- difftime(as.Date("2023-06-01"), as.Date(airbnb$host_since), units = "days")
airbnb$host_experience <- as.double(airbnb$host_experience)/365
airbnb <- airbnb[complete.cases(airbnb[, c("host_experience")]),]

airbnb$entire_apt <- ifelse(airbnb$room_type == "Entire home/apt", 1, 0)

airbnb$host_is_superhost <- ifelse( (airbnb$host_response_rate >=90 & airbnb$number_of_reviews >=10 & airbnb$review_scores_rating >= 4.8 ), 1, 0)

airbnb <- airbnb[complete.cases(airbnb[, c("host_is_superhost")]),]
airbnb <- airbnb[order(airbnb$id),]

# create train and test sets
train_indices <- sample(1:nrow(airbnb), 
                        size = ceiling(nrow(airbnb)*0.9),
                        replace = FALSE)
test_indices <- setdiff(1:nrow(airbnb), train_indices)

# create model
model.mat <- model.matrix(price ~ (accommodates + beds + host_experience +  number_of_reviews + review_scores_rating)^2 + host_is_superhost + entire_apt +
                            I(accommodates^2) + I(beds^2) + I(host_experience^2) + I(number_of_reviews^2) + I(review_scores_rating^2), 
                          data = airbnb)
model_df <- as.data.frame(model.mat)
model_df$`(Intercept)` = NULL
model_df$price = airbnb$price


# 2(a)
pca_model <- prcomp(model_df, center = TRUE, scale. = TRUE)
pve <- summary(pca_model)$importance[2, 1:4]
cum_pve <- cumsum(pve)

cat("Proportion of Variance Explained by each of the first 4 PCs:", pve, "\n")
cat("Cumulative Proportion of Variance Explained by the first 4 PCs:", cum_pve, "\n")

# 2(b)
top4_pc <- pca_model$x[, 1:4]
model_df$PC1 <- top4_pc[, 1]
model_df$PC2 <- top4_pc[, 2]
model_df$PC3 <- top4_pc[, 3]
model_df$PC4 <- top4_pc[, 4]

airbnb_train <- model_df[train_indices, ]
airbnb_test <- model_df[test_indices, ]
price = airbnb$price

# PC model
pc_model <- lm(price ~ PC1 + PC2 + PC3 + PC4, data = airbnb_train)
predictions_pc <- predict(pc_model, newdata = airbnb_test)
mse_pc <- mse(price, predictions_pc)

# polynomial model
poly_model <- lm(price ~ . -PC1 -PC2 -PC3 -PC4, data = airbnb_train)
predictions_poly <- predict(poly_model, newdata = airbnb_test)
mse_poly <- mse(price, predictions_poly)

# lasso model
lasso.reg <- glmnet(model.mat[train_indices, ], price[train_indices], 
                    alpha = 1, thresh = 1e-12)
lasso.cv <- cv.glmnet(model.mat[train_indices, ], price[train_indices], alpha = 1, nfolds = 10)
lambda_hat <- lasso.cv$lambda.min
lasso.cv.pred <- predict(lasso.reg, s = lambda_hat, newx = model.mat[test_indices,])
mse_lasso <- mse(price[test_indices], lasso.cv.pred)

# Ridge
ridge.reg <- glmnet(model.mat[train_indices,], price[train_indices], alpha = 0, thresh = 1e-12)
ridge.cv <- cv.glmnet(model.mat[train_indices, ], price[train_indices], alpha = 0, nfolds = 10)
lambda_hat <- ridge.cv$lambda.min
ridge.cv.pred <- predict(ridge.reg, s = lambda_hat, newx = model.mat[test_indices,])
mse_ridge <- mse(price[test_indices], ridge.cv.pred)

mse_results2 <- data.frame(
  PCA = mse_pc,
  Polynomial = mse_poly,
  Ridge = mse_ridge,
  Lasso = mse_lasso
)

print(mse_results2)

#-------------------------------------------------------------------------------------------#

# Clear the environment
rm(list=ls())

# Load necessary libraries
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(stats)

# ---------------------------------
# Question 3
# ---------------------------------

# Clean data for superhost prediction
airbnb_data <- read.csv("./Downloads/airbnb_data.csv")
airbnb_data$X <- NULL

datam <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating")]),]
datam$host_experience <- difftime(as.Date("2023-06-01"), as.Date(datam$host_since), units = "days")
datam$host_experience <- as.double(datam$host_experience)/365
datam <- datam[complete.cases(datam[, c("host_experience")]),]

datam$entire_apt <- ifelse(datam$room_type == "Entire home/apt", 1, 0)

datam$host_is_superhost <- ifelse( (datam$host_response_rate >=90 & datam$number_of_reviews >=10 & datam$review_scores_rating >= 4.8 ), 1, 0)

datam <- datam[complete.cases(datam[, c("host_is_superhost")]),]
datam <- datam[order(datam$id),]
datam <- datam[complete.cases(datam[, c("host_is_superhost")]),]
datam <- datam[order(datam$id),]

# Preliminary data cleaning:
datam$host_identity_verified <- ifelse(datam$host_identity_verified == "t", 1, 0)
datam <- datam[complete.cases(datam[, c("review_scores_rating", "review_scores_accuracy", "review_scores_value")]),]
#datam$host_is_superhost <- as.factor(datam$host_is_superhost)


# analysis
set.seed(0)
train <- sample(1:nrow(datam), .9*nrow(datam))
test <- (-train)  
y_test <- datam$host_is_superhost[test]
data_train <- datam[train, ]

# Cross-validated flexible logit model
x <- model.matrix(host_is_superhost ~  (review_scores_rating + host_experience + review_scores_accuracy + beds + review_scores_value)^2 + 
                    I(review_scores_rating^2) + I(host_experience^2) + I(review_scores_accuracy^2) + I(beds^2) + I(review_scores_value^2),  data = datam)
y <- datam$host_is_superhost
lasso.logit <- cv.glmnet(x[train,] , y[train], alpha = 1, family = binomial(link = "logit"))
lambda_hat <- lasso.logit$lambda.min
data_train$logit_lasso_pred <- predict(lasso.logit, s = lambda_hat, x[train,], type = "response")
test_lasso_logit <- mean((predict(lasso.logit, s = lambda_hat, x[test, ], type = "response") >= .5) != y_test)

# Random forest
forest_1 = randomForest(host_is_superhost ~  (review_scores_rating + host_experience + review_scores_accuracy + beds + review_scores_value)^2 + 
                          I(review_scores_rating^2) + I(host_experience^2) + I(review_scores_accuracy^2) + I(beds^2) + I(review_scores_value^2),
                        data = data_train,
                        ntree = 1000,
                        mtry = 4)
forest_1_pred <- predict(forest_1, data_train, type = "response")
test_forest <- mean((predict(forest_1, datam[test,], type = "response") >= 0.5) != y_test)

# MCE comparison
mce_values <- data.frame(L1_Regularized = test_lasso_logit, Random_forest = test_forest)
mce_table <- knitr::kable(mce_values, caption = "MCE Values")
print(mce_table)

#3b.
# Select relevant variables for clustering
cluster_vars <- data_train %>%
  select(host_is_superhost, review_scores_rating, host_experience)

# Scale the variables
scaled_cluster_vars <- scale(cluster_vars)

# Perform K-means clustering on the training data
kmeans_result <- kmeans(scaled_cluster_vars, centers = 1000)

# Add cluster assignment to the training data
data_train$cluster <- kmeans_result$cluster

# Mode function
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Summarize the training data by cluster
summary_data <- data_train %>%
  group_by(cluster) %>%
  summarize(
    avg_host_is_superhost = Mode(host_is_superhost), # Take mode for host_is_superhost
    avg_review_scores_rating = mean(review_scores_rating),
    avg_host_experience = mean(host_experience)
  )

# Pairwise correlations in the training data
cor_training <- cor(data_train[, c("host_is_superhost", "review_scores_rating", "host_experience")])

# Pairwise correlations in the summary data
cor_summary <- cor(summary_data[, c("avg_host_is_superhost", "avg_review_scores_rating", "avg_host_experience")])

# Display correlations
print("Pairwise correlations in the training data:")
print(cor_training)

print("Pairwise correlations in the summary data:")
print(cor_summary)

#Performing SVM on the summarized data
summary_train <- sample(1:nrow(summary_data), .9*nrow(summary_data))
summary_test <- (-summary_train)  
summary_y_test <- summary_data$avg_host_is_superhost[summary_test]

summary_data_train <- summary_data[summary_train, c("avg_host_is_superhost", "avg_review_scores_rating", "avg_host_experience")]
summary_data_train$avg_host_is_superhost <- as.factor(ifelse(summary_data_train$avg_host_is_superhost == 0, -1, 1))

tune.out2 <- tune(svm, avg_host_is_superhost ~  ., data = summary_data_train, kernel="radial", scale = TRUE, gamma = 0.01, 
                  ranges = list(cost = c(1, 10, 100, 10^3, 10^4)))

svm.pred2 <- predict(tune.out2$best.model, newdata = summary_data[summary_test, c("avg_review_scores_rating", "avg_host_experience")])
svm.pred2 <- ifelse(svm.pred2 == -1 , 0 , 1)

#Performing SVM on unclustered data
data_train <- datam[train, c("host_is_superhost", "review_scores_rating", "host_experience")]
data_train$host_is_superhost <- as.factor(ifelse(data_train$host_is_superhost == 0, -1, 1))

tune.out1 <- tune(svm, host_is_superhost ~  ., data = data_train, kernel="radial", scale = TRUE, gamma = 0.01, 
                  ranges = list(cost = c(1, 10, 100, 10^3, 10^4)))

svm.pred1 <- predict(tune.out1$best.model, newdata = datam[test, c("review_scores_rating", "host_experience")])
svm.pred1 <- ifelse(svm.pred1 == -1 , 0 , 1)

#Comparing MCE values
test_svm1 <- mean(svm.pred1 != y_test)
test_svm2 <- mean(svm.pred2 != summary_y_test)

mce_values_2 <- data.frame(SVM = test_svm1, SVM_clustered = test_svm2)
mce_table_2 <- knitr::kable(mce_values_2, caption = "SVM MCE Values =")
print(mce_table_2)