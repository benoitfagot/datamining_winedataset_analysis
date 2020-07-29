library(ggplot2)
library(plyr)
library(caret)
library(reshape2)
library(MASS)
library(ggcorrplot)
library(plotmo)
library(caret)

#Decision tree package
library(rpart)
library(rpart.plot)
library(e1071)
library(CatEncoders)
library(class)


data = read.csv("C:/Users/ucp/Desktop/workspace/R/projet_DM/wine.data", header = FALSE)
str(data)
colnames(data) <- c("type","alcohol", "malic_acid","ash",
                   "alcalinity_of_ash","magnesium","total_phenols",
                   "flavanoids","nonflavanoid_phenol",
                   "proanthocyanins","color_intensity","hue","id_of_diluted_wine","proline")

data$type <- as.factor(data$type)

str(data)


# statistiques de chaque colonnes
summary(data)

# check for missing values
for (i in colnames(data)){
  print(paste(i,':', sum(is.na(data[[i]])),'missing values'))
}

#check distinct values
#reminder : 178 objects
for (i in colnames(data)){
  print(paste(i,':', length(unique(data[[i]])),'distinct values'))
}


oldpar = par(mfrow = c(2,7))
for ( i in 1:14 ) {
  boxplot(dat[[i]])
  mtext(names(data)[i], cex = 0.8, side = 1, line = 2)
}
par(oldpar)

pairs(dat[, -grep("type", colnames(data))])

oldpar = par(mfrow = c(2,7))
for ( i in 1:14 ) {
  truehist(data[[i]], xlab = names(data)[i], col = 'lightgreen', main = paste("Average =", signif(mean(data[[i]]),3)))
}

data$type <- as.factor(data$type)

str(data)



###### on observe qu'un alcohol < 12.3 est forcement de type 2, cela pourrait aider l'arbre de d?cision
qplot(malic_acid, type, data = data)
##### ici on peut conjecturer que les acides maliques > 4.5 sont de type 3 malgr? la pr?sence d'un outlier de type2,
##### on ?met plus de r?serve sur la pertinence de cet attribut

qplot(ash, type, data = data)
##### ici on peut observer qu'un taux d'ash < 2.0 permet d'identifier des types 2, peut ?tre utile pour d?tecter ce type
##### cependant il n'est pas un bon indicateur pour les 2 autres types

qplot(alcalinity_of_ash, type, data = data)
##### alcalinity of ash pourrait nous aider ? identifier quelques alcohols du type 1 mais reste trop commun aux autres types

qplot(magnesium, type, data = data)
#### la grande majorit? des valeurs de magnesium se trouvent dans l'itnervalle [80,130] et ne permet pas de prendre de d?cision
#### cet attribut est tr?s s?rement ? ?carter

qplot(total_phenols, type, data = data)
##### cet attribut est plut?t bien r?parti pour prendre une d?cision pour les types 1 et 3, et on a vu pr?cedemment
##### que le type 2 ?tait identifiable par 2 autres attributs

qplot(flavanoids, type, data = data)
#### l? encore cet attribut s?pare bien les types 1 et 3

qplot(nonflavanoid_phenol, type, data = data)
#### nonflavanoid_phenol ne permet pas d'identifier correctement les types, trop commun

qplot(proanthocyanins, type, data = data)
#### peut permettre de trancher entre type 1/3

qplot(color_intensity, type, data = data)
#### cet attribut donne une bonne indication pour le type 2

qplot(hue, type, data = data)
#### cet attribut peut permettre de trancher entre les 3 types m?me s'il est pas tr?s r?parti

qplot(id_of_diluted_wine, type, data = data)
### cet attribut d?coupe parfaitement les type 1 et 3

qplot(proline, type, data = data)
#### cet attribut d?coupe tr?s bien les trois types et peut ?tre un bon attribut pour l'arbre



### nous retenons color_intensity, flavanoids, id_of_diluted_wine, proline comme attributs pour la d?cision
data2 <- subset(data, select=c("type", "color_intensity", "flavanoids", "id_of_diluted_wine","proline" ))
data2
pairs(data2[, -grep("type", colnames(data2))])

###################---------------ARBRE DE DECISION---------------------##############################
set.seed(12345) 
ind <- sample(2, nrow(data2), replace=TRUE, prob=c(0.7, 0.3))
data_train  <- data2[ind==1,]
data_test  <- data2[ind==2,]


library(rpart)
library(rpart.plot)
fit <- rpart(type~., data = data_train, method = 'class')
rpart.plot(fit, extra = 104)


predict_unseen <-predict(fit, data_test, type = 'class')

table_mat <- table(data_test$type, predict_unseen)
table_mat
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))

accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, data_test, type = 'class')
  table_mat <- table(data_test$type, predict_unseen)
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  accuracy_Test
}

control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 3,
                         cp = 0)
tune_fit <- rpart(type~., data = data_train, method = 'class', control = control)
accuracy_tune(tune_fit)


########################-------------kNN classifier using caret------------########################

# Configuration des commandes de train
repeats = 3
numbers = 10
tunel = 20


df_knn <- createDataPartition(data2$type, p = 0.7, list = FALSE)
df_knn
df2_train <- data2[df_knn, ]
df2_test <- data2[-df_knn, ]

trainCtrl <- trainControl(method = "repeatedcv", number = numbers, repeats = repeats)
#KNN en utilisant la m?thode du train fret caret
model_knn <- train(type ~., data = df2_train, method = "knn", 
                   trControl = trainCtrl, 
                   preProcess = c("center", "scale"),  
                   tuneLength = tunel )
model_knn
plot(model_knn)

varImp(model_knn)


prediction_knn <- predict(model_knn, newdata = df2_test)
confusionMatrix(prediction_knn, reference = df2_test$type)

