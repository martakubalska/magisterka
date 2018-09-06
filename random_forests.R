#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------


####------------------------------------------------------------------------####
#                                  RANDOM FORESTS
####------------------------------------------------------------------------####

source("./load_libraries.R")

cores <- detectCores() - 1 

####---------------------------------ON PCA---------------------------------####

train_pca_u <- train_pca[,1:104]
test_pca_u <- test_pca[,1:104]


selected_fold_number <- 5
selected_tune_length <- 26
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)


ctrl <- trainControl(method = "cv",
                     number = 5,
                     returnResamp="all",
                     summaryFunction = twoClassSummary,
                     search = "grid",
                     classProbs = T,
                     sampling = "down",
                     savePredictions = "final",
                     allowParallel = T,
                     seeds = seeds,
                     index = createFolds(y = train_pca_u$target, 
                                         k = selected_fold_number))

t0 <- Sys.time()

cl <- makeCluster(cores)
registerDoParallel(cl)
model_randomforest <- train(target ~ ., 
                           data = train_pca_u,
                           method = "rf", 
                           metric = "ROC",
                           trControl = ctrl,
                           tuneGrid = expand.grid(mtry = seq(5,30,by = 1)))
stopCluster(cl)

t1 <- Sys.time()


predict_test <- predict(model_randomforest, newdata = test_pca_u, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_pca_u$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )



####----------------------------------ON FULL DATA---------------------------####
selected_fold_number <- 5
selected_tune_length <- 11
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)


#sqrt z liczby kolumn model matrix okolo 28

ctrl <- trainControl(method = "cv",
                     number = 5,
                     returnResamp="all",
                     summaryFunction = twoClassSummary,
                     search = "grid",
                     classProbs = T,
                     sampling = "down",
                     savePredictions = "final",
                     allowParallel = T,
                     seeds = seeds,
                     index = createFolds(y = train$target, 
                                         k = selected_fold_number))

t0 <- Sys.time()

cl <- makeCluster(cores)
registerDoParallel(cl)
model_randomforest <- train(target ~ ., 
                            data = train,
                            method = "rf", 
                            metric = "ROC",
                            trControl = ctrl,
                            tuneGrid = expand.grid(mtry = seq(24,44,by = 2)))
stopCluster(cl)

t1 <- Sys.time()


predict_test <- predict(model_randomforest, newdata = test, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve



####----------------------------------ON SIMULATED ANNEALING---------------------------####
selected_fold_number <- 5
selected_tune_length <- 13
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)

#model matrix: 366 kolumn, sqrt: 19

ctrl <- trainControl(method = "cv",
                     number = 5,
                     returnResamp="all",
                     summaryFunction = twoClassSummary,
                     search = "grid",
                     classProbs = T,
                     sampling = "down",
                     savePredictions = "final",
                     allowParallel = T,
                     seeds = seeds,
                     index = createFolds(y = train_sa$target, 
                                         k = selected_fold_number))

t0 <- Sys.time()

cl <- makeCluster(cores)
registerDoParallel(cl)
model_randomforest_SA <- train(target ~ ., 
                            data = train_sa,
                            method = "rf", 
                            metric = "ROC",
                            trControl = ctrl,
                            tuneGrid = expand.grid(mtry = seq(11,35,by = 2)))
stopCluster(cl)

t1 <- Sys.time()

predict_test <- predict(model_randomforest_SA, newdata = test_sa, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_sa$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
