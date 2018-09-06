#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------
####                              LOGITS                                      ####
#--------------------------------------------------------------------------------
source("./load_libraries.R")

cores <- detectCores() - 1 


####-----------------------------LASSO----------------------------------------####

selected_fold_number <- 5
selected_tune_length <- 100
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)

myControl <- trainControl(method="cv", number=5, returnResamp="all",
                          classProbs=TRUE, savePredictions = "final",
                          summaryFunction=twoClassSummary,
                          sampling = "down",
                          allowParallel = T,
                          seeds = seeds,
                          index = createFolds(y = train$target, 
                                              k = selected_fold_number))
cl <- makeCluster(cores)
registerDoParallel(cl)
model_lasso <- train(target ~ ., data = train,
                     method = "glmnet", 
                     trControl = myControl,
                     metric = "ROC",
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = seq(0.0005,0.05,by = 0.0005)))
stopCluster(cl)


plot(model_lasso) #ROC od lambda

model_lasso$bestTune #best lambda

sum(coef(model_lasso$finalModel, model_lasso$bestTune$lambda)[,1] != 0) #variables used

#predictions
predict_test <- predict(model_lasso, newdata = test, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


####-----------------------------FORWARD SELECTION----------------------------####
set.seed(1994)

#str_sub(paste0(names(train)[-1]," + ", collapse = " "),start = 1,end=-4)

train_down <- downSample(train[,-1],train[,1], list=FALSE, yname="target")
test_down <- downSample(test[,-1],test[,1], list=FALSE, yname="target")

m <- glm(target ~ 1, family=binomial, data = train_down)

model_forward <- stepAIC(m, direction="forward", 
                         scope=list(lower=m,upper=~ VAR_0001 +  VAR_0005 +  VAR_0096 +  VAR_0101 +  VAR_0102 +  VAR_0111 +  VAR_0112 +  VAR_0117 +  
                                      VAR_0170 +  VAR_0187 +  VAR_0237 +  VAR_0260 +  VAR_0285 +  VAR_0287 +  VAR_0306 +  VAR_0351 +  
                                      VAR_0352 +  VAR_0357 +  VAR_0362 +  VAR_0383 +  VAR_0400 +  VAR_0405 +  VAR_0482 +  VAR_0503 +  
                                      VAR_0504 +  VAR_0505 +  VAR_0520 +  VAR_0548 +  VAR_0606 +  VAR_0668 +  VAR_0733 +  VAR_0734 +  
                                      VAR_0735 +  VAR_0737 +  VAR_0745 +  VAR_0759 +  VAR_0761 +  VAR_0762 +  VAR_0763 +  VAR_0765 +  
                                      VAR_0767 +  VAR_0772 +  VAR_0773 +  VAR_0775 +  VAR_0777 +  VAR_0784 +  VAR_0786 +  VAR_0803 +  
                                      VAR_0808 +  VAR_0928 +  VAR_0932 +  VAR_0941 +  VAR_0947 +  VAR_0948 +  VAR_0953 +  VAR_0960 +  
                                      VAR_0965 +  VAR_0967 +  VAR_0971 +  VAR_0972 +  VAR_1098 +  VAR_1099 +  VAR_1100 +  VAR_1101 +  
                                      VAR_1102 +  VAR_1103 +  VAR_1104 +  VAR_1105 +  VAR_1106 +  VAR_1107 +  VAR_1186 +  VAR_1188 +  
                                      VAR_1189 +  VAR_1190 +  VAR_1196 +  VAR_1197 +  VAR_1205 +  VAR_1206 +  VAR_1207 +  VAR_1218 +  
                                      VAR_1253 +  VAR_1255 +  VAR_1256 +  VAR_1257 +  VAR_1324 +  VAR_1325 +  VAR_1326 +  VAR_1377 +  
                                      VAR_1378 +  VAR_1379 +  VAR_1394 +  VAR_1400 +  VAR_1401 +  VAR_1405 +  VAR_1406 +  VAR_1407 +  
                                      VAR_1408 +  VAR_1414 +  VAR_1416 +  VAR_1422 +  VAR_1423 +  VAR_1505 +  VAR_1506 +  VAR_1507 +  
                                      VAR_1508 +  VAR_1509 +  VAR_1510 +  VAR_1511 +  VAR_1534 +  VAR_1535 +  VAR_1538 +  VAR_1539 +  
                                      VAR_1547 +  VAR_1548 +  VAR_1549 +  VAR_1563 +  VAR_1564 +  VAR_1586 +  VAR_1587 +  VAR_1588 +  
                                      VAR_1594 +  VAR_1595 +  VAR_1596 +  VAR_1597 +  VAR_1603 +  VAR_1607 +  VAR_1632 +  VAR_1633 +  
                                      VAR_1634 +  VAR_1635 +  VAR_1636 +  VAR_1637 +  VAR_1638 +  VAR_1660 +  VAR_1663 +  VAR_1671 +  
                                      VAR_1672 +  VAR_1673 +  VAR_1674 +  VAR_1675 +  VAR_1676 +  VAR_1677 +  VAR_1678 +  VAR_1679 +  
                                      VAR_1680 +  VAR_1681 +  VAR_1682 +  VAR_1683 +  VAR_1702 +  VAR_1703 +  VAR_1704 +  VAR_1705 +  
                                      VAR_1706 +  VAR_1707 +  VAR_1708 +  VAR_1749 +  VAR_1816 +  VAR_1817 +  VAR_1818 +  VAR_1819 +  
                                      VAR_1820 +  VAR_1821 +  VAR_1822 +  VAR_1862 +  VAR_1863 +  VAR_1885 +  VAR_1896 +  VAR_1897 +  
                                      VAR_1905 +  VAR_1916 +  VAR_1917 +  VAR_1934 +  VAR_0097 +  VAR_0241 +  VAR_0245 +  VAR_0263 +  
                                      VAR_0272 +  VAR_0304 +  VAR_0358 +  VAR_0540 +  VAR_0557 +  VAR_0559 +  VAR_0560 +  VAR_0575 +  
                                      VAR_0587 +  VAR_0614 +  VAR_0651 +  VAR_0693 +  VAR_0729 +  VAR_0742 +  VAR_0753 +  VAR_0771 +  
                                      VAR_0783 +  VAR_0792 +  VAR_0823 +  VAR_0824 +  VAR_0832 +  VAR_0850 +  VAR_0852 +  VAR_0855 +  
                                      VAR_0876 +  VAR_0881 +  VAR_0882 +  VAR_0886 +  VAR_0905 +  VAR_0910 +  VAR_0926 +  VAR_0929 +  
                                      VAR_0942 +  VAR_0944 +  VAR_0945 +  VAR_0961 +  VAR_0962 +  VAR_0964 +  VAR_1021 +  VAR_1080 +  
                                      VAR_1135 +  VAR_1209 +  VAR_1258 +  VAR_1260 +  VAR_1263 +  VAR_1397 +  VAR_1409 +  VAR_1421 +  
                                      VAR_1452 +  VAR_1527 +  VAR_1529 +  VAR_1536 +  VAR_1540 +  VAR_1541 +  VAR_1550 +  VAR_1649 +  
                                      VAR_1713 +  VAR_1742 +  VAR_1743 +  VAR_1752 +  VAR_1856 +  VAR_1864 +  VAR_0002 +  VAR_0003 +  
                                      VAR_0004 +  VAR_0016 +  VAR_0037 +  VAR_0049 +  VAR_0053 +  VAR_0129 +  VAR_0141 +  VAR_0224 +  
                                      VAR_0231 +  VAR_0293 +  VAR_0361 +  VAR_0364 +  VAR_0454 +  VAR_0512 +  VAR_0535 +  VAR_0537 +  
                                      VAR_0542 +  VAR_0550 +  VAR_0561 +  VAR_0615 +  VAR_0618 +  VAR_0621 +  VAR_0624 +  VAR_0625 +  
                                      VAR_0627 +  VAR_0698 +  VAR_0722 +  VAR_0730 +  VAR_0795 +  VAR_0816 +  VAR_0820 +  VAR_0828 +  
                                      VAR_0909 +  VAR_0919 +  VAR_0949 +  VAR_0954 +  VAR_0966 +  VAR_0969 +  VAR_0973 +  VAR_1007 +  
                                      VAR_1022 +  VAR_1027 +  VAR_1041 +  VAR_1058 +  VAR_1108 +  VAR_1115 +  VAR_1117 +  VAR_1118 +  
                                      VAR_1119 +  VAR_1122 +  VAR_1132 +  VAR_1149 +  VAR_1161 +  VAR_1173 +  VAR_1183 +  VAR_1233 +  
                                      VAR_1384 +  VAR_1402 +  VAR_1465 +  VAR_1528 +  VAR_1556 +  VAR_1571 +  VAR_1573 +  VAR_1651 +  
                                      VAR_1652 +  VAR_1653 +  VAR_1661 +  VAR_1711 +  VAR_1715 +  VAR_1754 +  VAR_1761 +  VAR_1826 +  
                                      VAR_1827 +  VAR_1830 +  VAR_1832 +  VAR_1853 +  VAR_0323 +  VAR_0806 +  VAR_0849 +  VAR_0853 +  
                                      VAR_0870 +  VAR_0885 +  VAR_0887 +  VAR_0891))



model_forward$coefficients %>% length() #267

#ROC
predict_test <- predict(model_forward, newdata = test, type = "response")
pred <- prediction(predict_test, test$target)
AUC <- (performance(pred,"auc")@y.values)[[1]]
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


####-----------------------------ON PCA---------------------------------------####
train_pca_u <- train_pca[,1:104]
test_pca_u <- test_pca[,1:104]

set.seed(1994)

myControl <- trainControl(method="cv", number=5,
                          classProbs=TRUE, savePredictions = "final",
                          summaryFunction=twoClassSummary,
                          sampling = "down")


model_PCA <- train(target ~.,data = train_pca_u,
                   method="glm",
                   family="binomial",
                   trControl = myControl,
                   metric = "ROC")

#Prediction on test set
predict_test <- predict(model_PCA, newdata = test_pca_u, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_pca_u$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


####------------------------------WITH SIMULATED ANNEALING---------------------####

#SIMULATED ANNEALING
set.seed(1994)

train_half <- sample_frac(train,0.5)

sa_ctrl <- safsControl(functions = rfSA,
                       method = "cv",
                       number = 5,
                       improve = 40,
                       allowParallel = TRUE)



cl <- makeCluster(cores)
registerDoParallel(cl)

rf_sa <- safs(x = train_half[,-1], y = train_half[,1],
              iters = 200,
              safsControl = sa_ctrl)

stopCluster(cl)


#CREATE NEW DATASET

final <- rf_sa$optVariables

train_sa <- train %>% select(one_of(final)) %>% bind_cols(train %>% select(target))
test_sa <- test %>% select(one_of(final)) %>% bind_cols(test %>% select(target))

#LOGIT ON NEW DATASET
set.seed(1994)

myControl <- trainControl(method="cv", number=5,
                          classProbs=TRUE, savePredictions = "final",
                          summaryFunction=twoClassSummary,
                          sampling = "down")


model_logit_sa <- train(target ~.,data = train_sa,
                   method="glm",
                   family="binomial",
                   trControl = myControl,
                   metric = "ROC")

#Prediction on test set
predict_test <- predict(model_logit_sa, newdata = test_sa, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_sa$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )



####------------------------------FULL---------------------------------------####

set.seed(1994)

myControl <- trainControl(method="cv", number=5,
                          classProbs=TRUE, savePredictions = "final",
                          summaryFunction=twoClassSummary,
                          sampling = "down")

cl <- makeCluster(cores)
registerDoParallel(cl)

model_logit_full <- train(target ~.,data = train,
                        method="glm",
                        family="binomial",
                        trControl = myControl,
                        metric = "ROC")

stopCluster(cl)

#Prediction on test set
predict_test <- predict(model_logit_full, newdata = test, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

