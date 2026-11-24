# Decision tree: predicting 10-year heart disease risk (Framingham)

framingham <- read.csv("data/framingham.csv")
str(framingham)
colSums(is.na(framingham))

# Remove rows with missing values (education, glucose, BMI, etc. have NAs)
framingham <- na.omit(framingham)
dim(framingham)
framingham$TenYearCHD <- as.factor(framingham$TenYearCHD)
table(framingham$TenYearCHD)
set.seed(123)

# Randomly select 70% of the row numbers for the training set
fr_index <- sample(1:nrow(framingham), size = 0.7 * nrow(framingham))

# Create the training and test datasets
fr_train <- framingham[fr_index, ]
fr_test  <- framingham[-fr_index, ]
dim(fr_train)
dim(fr_test)


library(rpart)       # rpart() - builds the decision tree
library(rpart.plot)  # rpart.plot() - draws the tree

# Only about 15% of people have heart disease, so a default tree just
# predicts "no" for everyone. The loss matrix makes missing a heart disease
# case (actual 1, predicted 0) 4 times as costly as a false alarm.
loss_matrix <- matrix(c(0, 4, 1, 0), nrow = 2)

# Build the decision tree on the training data
tree_model <- rpart(TenYearCHD ~ ., data = fr_train, method = "class",
                    parms = list(loss = loss_matrix),
                    control = rpart.control(cp = 0.01))

# Print the tree rules
print(tree_model)

printcp(tree_model)

# Variable importance (which variables matter most)
round(tree_model$variable.importance, 1)

# Plot the decision tree
rpart.plot(tree_model, type = 2, extra = 104, under = TRUE,
           main = "Decision Tree - 10-Year CHD Risk (Framingham)")


# Predict the class (0 or 1) for each person in the test data
tree_pred <- predict(tree_model, newdata = fr_test, type = "class")

# Confusion matrix: rows = actual, columns = predicted
tree_cm <- table(Actual = fr_test$TenYearCHD, Predicted = tree_pred)
tree_cm

TN <- tree_cm[1, 1]   # actual 0, predicted 0
FP <- tree_cm[1, 2]   # actual 0, predicted 1
FN <- tree_cm[2, 1]   # actual 1, predicted 0
TP <- tree_cm[2, 2]   # actual 1, predicted 1

# Calculate the metrics
accuracy    <- (TP + TN) / sum(tree_cm)
recall      <- TP / (TP + FN)                               
specificity <- TN / (TN + FP)
precision   <- TP / (TP + FP)
f1_score    <- 2 * precision * recall / (precision + recall)
error_rate  <- 1 - accuracy

# Show all the metrics rounded to 3 decimals
tree_metrics <- c(Accuracy = accuracy, Recall = recall,
                  Specificity = specificity, Precision = precision,
                  F1_Score = f1_score, Error_Rate = error_rate)
round(tree_metrics, 3)
