# Load necessary libraries
library(dplyr)
library(caret)
library(e1071)
library(kernlab)
library(glmnet)
library(doParallel)

# Set working directory
setwd("/Users/viditpokharna/Downloads")

# 1. Preliminary Data Cleaning
airbnb_data <- read.csv("ps3.csv")
airbnb_data$X <- NULL
airbnb_data$host_identity_verified = ifelse(airbnb_data$host_identity_verified == 't', 1, 0)
datam <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating", "review_scores_accuracy", "review_scores_value")]),]

# Data Cleaning
datam$host_experience <- difftime(as.Date("2023-06-01"), as.Date(datam$host_since), units = "days")
datam$host_experience <- as.double(datam$host_experience)/365
datam <- datam[complete.cases(datam[, c("host_experience")]),]
datam$entire_apt <- ifelse(datam$room_type == "Entire home/apt", 1, 0)
datam$host_is_superhost <- ifelse( (datam$host_response_rate >=90 & datam$number_of_reviews >=10 & datam$review_scores_rating >= 4.8 ), 1, 0)
datam <- datam[complete.cases(datam[, c("host_is_superhost")]),]
datam <- datam[order(datam$id),]

# 2. Analysis:
# (a) Allocate 10% observations to a test set
set.seed(0)
data_test = datam[sample(1:nrow(datam), 0.1*nrow(datam)), ]
data_train = anti_join(datam, data_test, by="id")

# (b) Linear probability model
lpm = lm(host_is_superhost ~ review_scores_rating, data=data_train)
lpm_predictions = ifelse(predict(lpm, data_test) > 0.5, 1, 0)
lpm_err = mean(lpm_predictions != data_test$host_is_superhost)

# (c) Logit model
mylogit = glm(host_is_superhost ~ review_scores_rating, data=data_train, family="binomial")
mylogit_predictions = ifelse(predict(mylogit, data_test, type="response") > 0.5, 1, 0)
mylogit_err = mean(mylogit_predictions != data_test$host_is_superhost)

# (d) Probit model
myprobit = glm(host_is_superhost ~ review_scores_rating, data=data_train, family=binomial(link="probit"))
myprobit_predictions = ifelse(predict(myprobit, data_test, type="response") > 0.5, 1, 0)
myprobit_err = mean(myprobit_predictions != data_test$host_is_superhost)

# (e) Coefficients from (b, c, d) 
lpm_coefficients <- coef(lpm)
logit_coefficients <- coef(mylogit)
probit_coefficients <- coef(myprobit)
coefficients_table <- data.frame(
  Model = c("Linear Probability", "Logit", "Probit"),
  Intercept = c(lpm_coefficients[1], logit_coefficients[1], probit_coefficients[1]),
  Review_Scores_Rating_Coefficient = c(lpm_coefficients[2], logit_coefficients[2], probit_coefficients[2])
)
print(coefficients_table)

# (f) SVM with a radial kernel
tune.out <- tune(svm, 
                 host_is_superhost ~ review_scores_rating + host_experience, 
                 data = data_train, 
                 kernel = "radial",
                 ranges = list(cost = c(1, 10, 100, 1e3, 1e4), gamma = 0.01))
mysvm = svm(host_is_superhost ~ review_scores_rating + host_experience,
            data = data_train,
            kernel = "radial",
            type = "C",
            gamma = 0.01,
            cost = tune.out$best.parameters$cost)
mysvm_predictions = predict(mysvm, data_test)
mysvm_predictions = ifelse(is.na(mysvm_predictions), 0, mysvm_predictions)
mysvm_err = mean(mysvm_predictions != data_test$host_is_superhost)

# (g) ℓ1 regularized logistic regression
model <- model.matrix(~ review_scores_rating 
                      + host_experience 
                      + review_scores_accuracy 
                      + beds 
                      + review_scores_value 
                      + review_scores_rating:host_experience 
                      + review_scores_rating:review_scores_accuracy
                      + review_scores_rating:beds
                      + review_scores_rating:review_scores_value
                      + host_experience:review_scores_accuracy
                      + host_experience:beds
                      + host_experience:review_scores_value
                      + review_scores_accuracy:beds
                      + review_scores_accuracy:review_scores_value
                      + beds:review_scores_value
                      + I(review_scores_rating^2)
                      + I(host_experience^2)
                      + I(review_scores_accuracy^2)
                      + I(beds^2)
                      + I(review_scores_value^2), 
                      data=data_train)
cv.lasso = cv.glmnet(model, data_train$host_is_superhost, alpha=1, family="binomial")
mylasso = glmnet(model, data_train$host_is_superhost, alpha=1, family="binomial", lambda=cv.lasso$lambda.min)

# (h) Mean classification error
model_test <- model.matrix(~ review_scores_rating 
                           + host_experience 
                           + review_scores_accuracy 
                           + beds 
                           + review_scores_value 
                           + review_scores_rating:host_experience 
                           + review_scores_rating:review_scores_accuracy
                           + review_scores_rating:beds
                           + review_scores_rating:review_scores_value
                           + host_experience:review_scores_accuracy
                           + host_experience:beds
                           + host_experience:review_scores_value
                           + review_scores_accuracy:beds
                           + review_scores_accuracy:review_scores_value
                           + beds:review_scores_value
                           + I(review_scores_rating^2)
                           + I(host_experience^2)
                           + I(review_scores_accuracy^2)
                           + I(beds^2)
                           + I(review_scores_value^2), 
                           data=data_test)

predictions <- predict(mylasso, newx = model_test)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
lasso_err <- mean(predicted_classes != data_test$host_is_superhost)
errors_table <- data.frame(
  Model = c("Linear Probability", "Logit", "Probit", "SVM", "Lasso"),
  Mean_Classification_Error = c(lpm_err, mylogit_err, myprobit_err, mysvm_err, lasso_err)
)

print(errors_table)