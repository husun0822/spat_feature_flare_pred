library(dplyr)
library(tidyr)
library(keras)
library(caret)
library(glmnet)
library(gglasso)
library(xgboost)
library(ggplot2)
library(fdapace)

GBoost = function(tt, feature){
  dtrain = tt$train
  dtest = tt$test
  
  xgtrain = xgb.DMatrix(data = as.matrix(dtrain[,feature]), 
                        label = ifelse(dtrain$fclass=="B",0,1))
  xgtest = as.matrix(dtest[,feature])
  xgb_model = xgboost(data = xgtrain,
                      max.depth = 2, eta = 1, nthread = 1, nrounds = 2, 
                      objective = "binary:logistic",verbose = 0)
  raw_predict = predict(xgb_model,xgtest)
  xgb_predict = ifelse(raw_predict > 0.5,1,0)
  
  cm = confusionMatrix(data = factor(xgb_predict, 
                                     levels=c(0,1), labels = c("B","M")), 
                       reference = dtest$fclass, positive = "M")
  cm_stat = data.frame(TP=0, FP=0, TN=0, FN=0, model="XGBoost")
  
  cm_stat[,c("TP","FP","TN","FN")] = c(cm$table[2,2], cm$table[2,1], 
                                       cm$table[1,1], cm$table[1,2])
  return(list(cm_stat=cm_stat,xgmodel = xgb_model,pred_score=raw_predict))
}
