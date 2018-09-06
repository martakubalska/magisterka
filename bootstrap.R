#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------



source("./load_libraries.R")


# generowanie bootstrapowych prob testowych
set.seed(1994)
bootstrap <- createDataPartition(test$target, times = 50, p = 0.5)


#### CUSTOM SUMMARY ####

custom_summary <- function(prognoza, dane){

  pred <- prediction(prognoza[,2], dane$target)
  
  rocAUC <-(performance(pred,"auc")@y.values)[[1]]
  
  x_acc <- (performance(pred,"acc")@x.values)[[1]]
  cutoff_acc=0.5
  n <- which(abs(x_acc-cutoff_acc)==min(abs(x_acc-cutoff_acc)))
  
  accuracy <- (performance(pred,"acc")@y.values)[[1]][n]
  
  
  x_lift <- (performance(pred,"lift")@x.values)[[1]]
  cutoff_lift=0.8
  n <- which(abs(x_lift-cutoff_lift)==min(abs(x_lift-cutoff_lift)))
  
  lift <- (performance(pred,"lift")@y.values)[[1]][n]
    
  wynik <- c(ROC = rocAUC,
             Accuracy = accuracy,
             Lift = lift)
  wynik
}



#### OCENA BOOTSTRAP ####

ocena_bootstrap <- function(zb_testowy, model){
  
  # generuje prognoze
  prognoza <- predict(model, zb_testowy, type = "prob")
  
  # opisowa ocena modelu
  ocena <- custom_summary(prognoza, zb_testowy)
  
  ocena
}

ocena_bootstrap(test,model_lasso)


#### WYNIKI BOOTSTRAP ####

wyniki_bootstrap <- function(df, model, proby_boot, nazwa_modelu){
  
  wynik_bootstrap <- data.frame(Model = rep(nazwa_modelu, length.out = length(proby_boot)),
                                ROC = numeric(length = length(proby_boot)), 
                                ACCURACY = numeric(length = length(proby_boot)), 
                                LIFT = numeric(length = length(proby_boot)),
                                stringsAsFactors = F)
  
  for(i in 1:length(proby_boot)){
    wynik_bootstrap[i,2:4] <- ocena_bootstrap(zb_testowy = df[proby_boot[[i]],], model = model)
  }
  
  
  temp <- tidyr::gather(wynik_bootstrap[,-1], key = "metric")
  srednie <- temp %>% group_by(metric) %>% summarise(srednia = round(mean(value), 4))
  temp <- temp %>% left_join(srednie)
  
   
  list(wyniki_df = wynik_bootstrap)
  
}