#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------


#####------------------------------------------------------------------------####
#                                 NEURAL NETWORKS
####-------------------------------------------------------------------------####


source("./load_libraries.R")

cores <- detectCores() - 1 

####---------------------------------ON PCA----------------------------------####

train_pca_u <- train_pca[,1:104]
test_pca_u <- test_pca[,1:104]


selected_fold_number <- 5
selected_tune_length <- 40
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)


parametry <- expand.grid(
  size = seq(1,10,1),
  decay = c(10^seq(-4, -1, 1))
)


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
model_neuralnet <- train(target ~ ., 
                            data = train_pca_u,
                            method = "nnet", 
                            metric = "ROC",
                            trControl = ctrl,
                            tuneGrid = parametry,
                         maxit = 500,
                         MaxNWts = 2000)
stopCluster(cl)

t1 <- Sys.time()

predict_test <- predict(model_neuralnet, newdata = test_pca_u, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_pca_u$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )



####--------------------------------WITH SIMULATED ANNEALING------------------------------####
selected_fold_number <- 5
selected_tune_length <- 40
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)


parametry <- expand.grid(
  size = seq(1,10,1),
  decay = c(10^seq(-4, -1, 1))
)


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


cl <- makeCluster(cores)
registerDoParallel(cl)
model_neuralnet_SA <- train(target ~ ., 
                            data = train_sa,
                            method = "nnet", 
                            metric = "ROC",
                            trControl = ctrl,
                            tuneGrid = parametry,
                            maxit = 750,
                            MaxNWts = 4000)
stopCluster(cl)


predict_test <- predict(model_neuralnet_SA, newdata = test_sa, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test_sa$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )


####--------------------------------ON FULL DATA------------------------------####
selected_fold_number <- 5
selected_tune_length <- 40
set.seed(1910)
seeds <- vector(mode = "list", length = selected_fold_number + 1)
for(i in 1:selected_fold_number) seeds[[i]] <- sample.int(10000, selected_tune_length)
seeds[[length(seeds)]] <- sample.int(10000, 1)


parametry <- expand.grid(
  size = c(1,3,5),
  decay = c(10^seq(-4, -1, 1))
)


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


cl <- makeCluster(cores)
registerDoParallel(cl)
model_neuralnet_full <- train(target ~ ., 
                              data = train,
                              method = "nnet", 
                              metric = "ROC",
                              trControl = ctrl,
                              tuneGrid = parametry,
                              maxit = 1000,
                              MaxNWts = 5000)
stopCluster(cl)


predict_test <- predict(model_neuralnet_full, newdata = test, type = "prob")

#Krzywa ROC
pred <- prediction(predict_test[,2], test$target)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc") # show calculated AUC for model
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
