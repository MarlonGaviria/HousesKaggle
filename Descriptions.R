# setwd('../../house-prices-advanced-regression-techniques/')
train <- read.csv('Datos/train.csv')
test <- read.csv('Datos/test.csv')

str(train)
dim(train)


str(test)
dim(test)

names(train)[!(names(train) %in% names(test))] # variable respuesta

apply(is.na(train),2,sum) # Cantidad de Na's por variable Train
apply(is.na(test),2,sum) # Cantidad de Na's por variabl Test

#ID's

trainId=train$Id
testId=test$Id
# eliminamos variables con muchos Na's
# train
train$Id <- NULL

mean(is.na(train$Alley));train$Alley <- NULL

mean(is.na(train$PoolQC));train$PoolQC <- NULL

mean(is.na(train$Fence));train$Fence <- NULL

mean(is.na(train$FireplaceQu));train$FireplaceQu <- NULL

mean(is.na(train$MiscFeature));train$MiscFeature <- NULL

mean(is.na(train$GarageYrBlt));train$GarageYrBlt <- NULL

#test
test$Id <- NULL

mean(is.na(test$Alley));test$Alley <- NULL

mean(is.na(test$PoolQC));test$PoolQC <- NULL

mean(is.na(test$Fence));test$Fence <- NULL

mean(is.na(test$FireplaceQu));test$FireplaceQu <- NULL

mean(is.na(test$MiscFeature));test$MiscFeature <- NULL

mean(is.na(test$GarageYrBlt));test$GarageYrBlt <- NULL

#data de entrenamiento (IMPUTACION)
# funcion para calcular la moda
mode <- function(v) {
	uniqv <- unique(v)
	uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Funcion para imputar una base de datos
fill.na <- function(dt){
	for(i in 1:ncol(dt)){
		if(class(dt[,i])=="character"){
			dt[,i][which(is.na(dt[,i]))] <- mode(dt[,i])
		}
		if(class(dt[,i])=="integer"){
			dt[,i][which(is.na(dt[,i]))] <- median(dt[,i],na.rm=T)
			
		}
	}
	return(dt)
}

train <- fill.na(train)
apply(is.na(train),2,sum)
sum(is.na(train)) # Cantidad de Na's por variable Train

test <- fill.na(test)
apply(is.na(test),2,sum) 
sum(is.na(test)) # Cantidad de Na's por variabl Test

# Modelos

# LR

LR <- lm(SalePrice~.,data=train)

mean(abs(LR$fitted.values-train$SalePrice))

predLR <- as.vector(predict(LR,test))

predLR <- data.frame(Id=testId,SalePrice=predLR)

write.csv(predLR,file = "Predicciones/PredLR.csv",row.names = F)

# RF
require(ranger)

RF <- ranger(SalePrice~.,data = na.omit(train),num.trees = 1500)

mean(abs(Rf$predictions-na.omit(train)$SalePrice))

predRF <- predict(RF,test)

predRF <- data.frame(Id=testId,SalePrice=predRF$predictions)

write.csv(predRf,file = "Predicciones/PredRF.csv",row.names = F)

# Knn

require(FNN)
# usamos solo las variables numericas
varsNum <- as.vector(which(sapply(train,is.numeric)==T))
knn <- knn.reg(train[,varsNum[-36]],test = test[,varsNum[-36]],y=train$SalePrice,k=6)

mean(abs(knn$pred-train$SalePrice))

predknn <- data.frame(Id=testId,SalePrice=knn$pred)

write.csv(predknn,file = "Predicciones/Predknn.csv",row.names = F)

# XGboost

require(xgboost)
require(fastDummies)
varsNumTt <- as.vector(which(sapply(test,is.numeric)==T))
trainxg <- train[,varsNum]
testinxg <- test[,varsNumTt]
i <- sample(1:nrow(trainxg),nrow(trainxg)*0.8)
xgtr <- trainxg[i,]
xgtt <- trainxg[-i,]

param1 <- list(booster = "gblinear", nthread = 100, eta = 0.1,
	       gamma = 0, max = 10, min_child_weight = 1,
	       max_delta_step = 2, lambda = 1,
	       alpha = 0, three_method = "approx",
	       objective = "reg:squarederror",
	       eval_metric = "mae")  # Parametros del Boosting


dtrain <- xgb.DMatrix(as.matrix(xgtr[,-36]),label = xgtr$SalePrice)
dtest <- xgb.DMatrix(as.matrix(xgtt[,-36]),label = xgtt$SalePrice)

watchlist <- list(train=na.omit(dtrain),test=na.omit(dtest))

XGboost <- xgb.train(data = na.omit(dtrain),nrounds = 1500,
			   watchlist = watchlist,params = param1,
			   max.depth=2)

predXGB <- predict(XGboost,as(as.matrix(testinxg),"dgCMatrix"))

predXGB <- data.frame(Id=testId,SalePrice=predXGB)

write.csv(predknn,file = "Predicciones/PredXGB.csv",row.names = F)
