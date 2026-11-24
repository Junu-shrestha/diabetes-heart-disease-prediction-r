# KNN classification: predicting diabetes and tuning k

library(caTools)
data.diabetes <- read.csv("data/diabetes.csv")
str(data.diabetes)
head(data.diabetes)
table(data.diabetes$Outcome)
set.seed(123)
split.data <- sample.split(data.diabetes$Outcome, SplitRatio = 0.5)
train.data <- subset(data.diabetes, split.data == TRUE)
test.data <- subset(data.diabetes, split.data == FALSE)
nrow(train.data)
nrow(test.data)

library(class)
library(caret)
train.scale <- scale(train.data[, 1:8])
test.scale <- scale(test.data[, 1:8],
                    center = attr(train.scale, "scaled:center"),
                    scale  = attr(train.scale, "scaled:scale"))
knn.k1  <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 1)
knn.k3  <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 3)
knn.k5  <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 5)
knn.k7  <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 7)
knn.k9  <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 9)
knn.k11 <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 11)
knn.k15 <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 15)
knn.k19 <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 19)
knn.k25 <- knn(train = train.scale, test = test.scale, cl = train.data$Outcome, k = 25)
knn.k5

pred.table <- rbind(
  "k=1"  = table(knn.k1),  "k=3"  = table(knn.k3),  "k=5"  = table(knn.k5),
  "k=7"  = table(knn.k7),  "k=9"  = table(knn.k9),  "k=11" = table(knn.k11),
  "k=15" = table(knn.k15), "k=19" = table(knn.k19), "k=25" = table(knn.k25))
pred.table

cm.k1  <- confusionMatrix(as.factor(knn.k1),  as.factor(test.data$Outcome))
cm.k3  <- confusionMatrix(as.factor(knn.k3),  as.factor(test.data$Outcome))
cm.k5  <- confusionMatrix(as.factor(knn.k5),  as.factor(test.data$Outcome))
cm.k7  <- confusionMatrix(as.factor(knn.k7),  as.factor(test.data$Outcome))
cm.k9  <- confusionMatrix(as.factor(knn.k9),  as.factor(test.data$Outcome))
cm.k11 <- confusionMatrix(as.factor(knn.k11), as.factor(test.data$Outcome))
cm.k15 <- confusionMatrix(as.factor(knn.k15), as.factor(test.data$Outcome))
cm.k19 <- confusionMatrix(as.factor(knn.k19), as.factor(test.data$Outcome))
cm.k25 <- confusionMatrix(as.factor(knn.k25), as.factor(test.data$Outcome))

accuracy.table <- data.frame(
  k = c(1, 3, 5, 7, 9, 11, 15, 19, 25),
  Accuracy = c(cm.k1$overall["Accuracy"],  cm.k3$overall["Accuracy"],
               cm.k5$overall["Accuracy"],  cm.k7$overall["Accuracy"],
               cm.k9$overall["Accuracy"],  cm.k11$overall["Accuracy"],
               cm.k15$overall["Accuracy"], cm.k19$overall["Accuracy"],
               cm.k25$overall["Accuracy"])
)
print(accuracy.table)
plot(accuracy.table$k, accuracy.table$Accuracy, type = "b",
     xlab = "Number of nearest neighbors (k)", ylab = "Accuracy",
     main = "KNN Accuracy for Different k Values")
best.k <- accuracy.table$k[which.max(accuracy.table$Accuracy)]
best.k
print(cm.k19)
