# Logistic regression vs. KNN for predicting diabetes

# Import the diabetes dataset
diabetes <- read.csv("data/diabetes.csv")
str(diabetes)
summary(diabetes)
head(diabetes)
diabetes$Outcome <- as.factor(diabetes$Outcome)
table(diabetes$Outcome)


set.seed(123)

# Randomly select 50% of the row numbers for the training set
train_index <- sample(1:nrow(diabetes), size = 0.5 * nrow(diabetes))

# Create the training and test datasets
train <- diabetes[train_index, ]
test  <- diabetes[-train_index, ]
dim(train)
dim(test)

logit_model <- glm(Outcome ~ ., data = train, family = binomial)
summary(logit_model)

# Odds ratios make the coefficients easier to interpret
exp(coef(logit_model))

# Predict the probability of diabetes for each person in the test data
test_prob <- predict(logit_model, newdata = test, type = "response")

# Convert probabilities to class predictions using a 0.5 threshold
test_pred <- ifelse(test_prob > 0.5, 1, 0)


logit_cm <- table(Actual = test$Outcome, Predicted = test_pred)
logit_cm

library(ROCR)

# Create a prediction object from the predicted probabilities and actual values
logit_rocr <- prediction(test_prob, test$Outcome)
logit_roc <- performance(logit_rocr, measure = "tpr", x.measure = "fpr")
plot(logit_roc, colorize = TRUE, lwd = 2,
     main = "ROC Curve - Logistic Regression (Diabetes)")
abline(a = 0, b = 1, lty = 2, col = "gray")

# Calculate the Area Under the Curve (AUC)
logit_auc <- performance(logit_rocr, measure = "auc")@y.values[[1]]
logit_auc

metric_at_05 <- function(pred_obj, measure) {
  perf <- performance(pred_obj, measure = measure)
  cutoffs <- perf@x.values[[1]]
  values  <- perf@y.values[[1]]
  idx <- which(cutoffs > 0.5)
  values[idx[length(idx)]]
}

# Calculate accuracy, recall, specificity, precision, F1 score and AUC
logit_metrics <- c(
  Accuracy    = metric_at_05(logit_rocr, "acc"),   
  Recall      = metric_at_05(logit_rocr, "tpr"),   
  Specificity = metric_at_05(logit_rocr, "tnr"),   
  Precision   = metric_at_05(logit_rocr, "prec"),  
  F1_Score    = metric_at_05(logit_rocr, "f"),     # balance of precision and recall
  AUC         = logit_auc                          # from part (c)
)


round(logit_metrics, 3)
plot(performance(logit_rocr, measure = "acc"),
     main = "Accuracy vs. Cutoff - Logistic Regression")


library(class)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))

predictors <- names(diabetes)[names(diabetes) != "Outcome"]
diabetes_norm <- as.data.frame(lapply(diabetes[, predictors], normalize))

# Use the same train/test rows as the logistic regression model
train_knn <- diabetes_norm[train_index, ]
test_knn  <- diabetes_norm[-train_index, ]
train_labels <- diabetes$Outcome[train_index]
test_labels  <- diabetes$Outcome[-train_index]

# k = 19 (about the square root of 384 training rows; odd number to avoid ties)
set.seed(123)
knn_pred <- knn(train = train_knn, test = test_knn, cl = train_labels,
                k = 19, prob = TRUE)

# Confusion matrix for KNN
knn_cm <- table(Actual = test_labels, Predicted = knn_pred)
knn_cm

# Convert the KNN vote proportion into the probability of diabetes (class 1)
knn_prob <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == "1", knn_prob, 1 - knn_prob)

# Use the same performance() function to calculate KNN metrics
knn_rocr <- prediction(knn_prob, test_labels)
knn_auc  <- performance(knn_rocr, measure = "auc")@y.values[[1]]

knn_metrics <- c(
  Accuracy    = metric_at_05(knn_rocr, "acc"),
  Recall      = metric_at_05(knn_rocr, "tpr"),
  Specificity = metric_at_05(knn_rocr, "tnr"),
  Precision   = metric_at_05(knn_rocr, "prec"),
  F1_Score    = metric_at_05(knn_rocr, "f"),
  AUC         = knn_auc
)
round(knn_metrics, 3)

# Plot both ROC curves on one chart to compare the models
knn_roc <- performance(knn_rocr, measure = "tpr", x.measure = "fpr")
plot(logit_roc, col = "blue", lwd = 2,
     main = "ROC Curve Comparison: Logistic Regression vs. KNN")
plot(knn_roc, col = "red", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Logistic Regression (AUC =", round(logit_auc, 3), ")"),
                  paste("KNN, k = 19 (AUC =", round(knn_auc, 3), ")")),
       col = c("blue", "red"), lwd = 2)

# Side-by-side comparison table of the two models
comparison <- data.frame(Logistic_Regression = round(logit_metrics, 3),
                         KNN = round(knn_metrics, 3))
comparison


