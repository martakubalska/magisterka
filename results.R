#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------


source("./load_libraries.R")

train_pca_u <- train_pca[,1:104]
test_pca_u <- test_pca[,1:104]

####-----------------------------------------BOOTSTRAP EVALUATION----------------####

eval_log_full <- wyniki_bootstrap(test, model_logit_full, bootstrap, "model_logit_full")
eval_lasso <- wyniki_bootstrap(test, model_lasso, bootstrap, "model_lasso")
eval_rf_full <- wyniki_bootstrap(test, model_randomforest, bootstrap, "model_randomforest_full")
eval_nn_full <- wyniki_bootstrap(test, model_neuralnet_full, bootstrap, "model_neuralnet_full")

eval_log_PCA <- wyniki_bootstrap(test_pca_u, model_PCA, bootstrap, "model_logit_PCA")
eval_rf_PCA <- wyniki_bootstrap(test_pca_u, model_randomforest, bootstrap, "model_randomforest_PCA")
eval_nn_PCA <- wyniki_bootstrap(test_pca_u, model_neuralnet, bootstrap, "model_neuralnet_PCA")

eval_log_SA <- wyniki_bootstrap(test_sa, model_logit_sa, bootstrap, "model_logit_SA")
eval_rf_SA <- wyniki_bootstrap(test_sa, model_randomforest_SA, bootstrap, "model_randomforest_SA")
eval_nn_SA <- wyniki_bootstrap(test_sa, model_neuralnet_SA, bootstrap, "model_neuralnet_SA")

eval_forward <- wyniki_bootstrap(test, model_forward, bootstrap, "model_forward")




full_eval <- eval_log_full %>% bind_rows(eval_lasso) %>% bind_rows(eval_rf_full) %>% 
  bind_rows(eval_nn_full) %>% bind_rows(eval_log_PCA) %>% bind_rows(eval_rf_PCA) %>%
  bind_rows(eval_nn_PCA) %>% bind_rows(eval_log_SA) %>% bind_rows(eval_rf_SA) %>%
  bind_rows(eval_nn_SA) %>% bind_rows(eval_forward)

metrics_avg <- full_eval %>% group_by(Model) %>% summarise(Roc = mean(ROC), Lift = mean(LIFT),
                                                  Accuracy = mean(ACCURACY))


####PLOTS
plot_roc <- ggplot(full_eval, aes(x=Model, y=ROC, fill = Model)) + 
  geom_boxplot() + theme(legend.position="none") + 
  scale_fill_manual(values=c("#FF9933", "#FFFF66","#CC0033", "#3399FF",
                             "#00CC66", "#CC0033","#3399FF", "#00CC66",
                             "#CC0033", "#3399FF","#00CC66")) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
plot_roc


plot_lift <- ggplot(full_eval, aes(x=Model, y=LIFT, fill = Model)) + 
  geom_boxplot() + theme(legend.position="none") + 
  scale_fill_manual(values=c("#FF9933", "#FFFF66","#CC0033", "#3399FF",
                            "#00CC66", "#CC0033","#3399FF", "#00CC66",
                            "#CC0033", "#3399FF","#00CC66")) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
plot_lift

plot_acc <- ggplot(full_eval, aes(x=Model, y=ACCURACY, fill = Model)) + 
  geom_boxplot() + theme(legend.position="none") + 
  scale_fill_manual(values=c("#FF9933", "#FFFF66","#CC0033", "#3399FF",
                             "#00CC66", "#CC0033","#3399FF", "#00CC66",
                             "#CC0033", "#3399FF","#00CC66"))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
plot_acc


####--------------------------------PREDICTIONS ON FULL TEST-------------------####

predictions <- data.frame(    logit_full = numeric(length = nrow(test)), 
                              lasso = numeric(length = nrow(test)), 
                              forward = numeric(length = nrow(test)),
                              logit_PCA = numeric(length = nrow(test)),
                              logit_SA = numeric(length = nrow(test)),
                              randomforest_full = numeric(length = nrow(test)),
                              randomforest_PCA = numeric(length = nrow(test)),
                              randomforest_SA = numeric(length = nrow(test)),
                              neuralnet_full = numeric(length = nrow(test)),
                              neuralnet_PCA = numeric(length = nrow(test)),
                              neuralnet_SA = numeric(length = nrow(test))
                              )



predictions[,1] <- predict(model_logit_full, newdata = test, type = "prob")[,2]
predictions[,2]<- predict(model_lasso, newdata = test, type = "prob")[,2]
predictions[,6] <- predict(model_randomforest, newdata = test, type = "prob")[,2]
predictions[,9] <- predict(model_neuralnet_full, newdata = test, type = "prob")[,2]

predictions[,4] <- predict(model_PCA, newdata = test_pca_u, type = "prob")[,2]
predictions[,7] <- predict(model_randomforest, newdata = test_pca_u, type = "prob")[,2]
predictions[,10] <- predict(model_neuralnet, newdata = test_pca_u, type = "prob")[,2]

predictions[,5] <- predict(model_logit_sa, newdata = test_sa, type = "prob")[,2]
predictions[,8] <- predict(model_randomforest_SA, newdata = test_sa, type = "prob")[,2]
predictions[,11] <- predict(model_neuralnet_SA, newdata = test_sa, type = "prob")[,2]

predictions[,3] <- predict(model_forward, newdata = test, type = "response")

cormat <- cor(predictions)

min(cormat)
cormat == max(cormat[cormat<1])

corrplot(cormat, method = "color", order="hclust")


corrplot2 <- function(corr) {
  a = 2 / (max(corr) - min(corr))
  b = 1 - (2 / (1 - (min(corr) / max(corr))))
  y = a * corr + b
  corrplot(y, method="circle", bg="grey92", 
           order="hclust", addrect=4, cl.lim=c(-1, 1), cl.pos = "n")
}

corrplot2(cormat)

####CONFUSSION MATRIX####
prediction <- predict(model_randomforest_SA, newdata = test_sa, type = "prob")

pred <- prediction(prediction[,2], test_sa$target)
perf <- performance(pred,"tpr","fpr")

prediction$prognose <- as.factor(ifelse(prediction$X1 >= 0.5, "X1", "X0"))
prediction$true <- test$target

table(prediction$prognose, prediction$true)

####WYKRES ROZKŁADOW####
ggplot(prediction, aes(x=X1, fill=true)) + geom_density(alpha=.4, size = 1) + 
  scale_fill_brewer(palette = "Set1")


####WYKRESY ROC####
prediction <- predict(model_neuralnet_full, newdata = test, type = "prob")

pred <- prediction(prediction[,2], test$target)
perf <- performance(pred,"tpr","fpr")
perf_lasso <- performance(pred,"tpr","fpr")
perf_rffull <- performance(pred,"tpr","fpr")
perf_nnfull <- performance(pred,"tpr","fpr")

df <- data.frame(
  fpr = perf@x.values %>% unlist(),
  tpr = perf@y.values %>% unlist(),
  model = rep("randomforest_SA", perf@y.values %>% unlist() %>% length())
)

df_lasso <- data.frame(
  fpr = perf_lasso@x.values %>% unlist(),
  tpr = perf_lasso@y.values %>% unlist(),
  model = rep("lasso", perf_lasso@y.values %>% unlist() %>% length())
)

df_rffull <- data.frame(
  fpr = perf_rffull@x.values %>% unlist(),
  tpr = perf_rffull@y.values %>% unlist(),
  model = rep("randomforest_full", perf_rffull@y.values %>% unlist() %>% length())
)

df_nnfull <- data.frame(
  fpr = perf_nnfull@x.values %>% unlist(),
  tpr = perf_nnfull@y.values %>% unlist(),
  model = rep("neuralnet_full", perf_nnfull@y.values %>% unlist() %>% length())
)

df %>% bind_rows(df_lasso) %>% bind_rows(df_rffull) %>% bind_rows(df_nnfull) %>%
  ggplot(aes(x = fpr, y = tpr, color = model)) + 
  geom_abline(intercept = 0, slope = 1, linetype = 2) +
  geom_line()

  