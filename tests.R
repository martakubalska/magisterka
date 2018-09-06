#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------


check_ROC <- function(model1, model2){
  p1 <- shapiro.test(full_eval$ROC[full_eval$Model == model1])$p.value
  p2 <- shapiro.test(full_eval$ROC[full_eval$Model == model2])$p.value
  
  if(p1 > 0.05 & p2 > 0.05){
    wynik <-  with(data = full_eval, expr = 
    {t.test(x = ROC[Model == model1], y = ROC[Model == model2], 
            alternative = "greater", mu = 0)})
  } else{
    wynik <- with(data = full_eval, expr = 
    {wilcox.test(x = ROC[Model == model1], y = ROC[Model == model2], 
                 alternative = "greater", mu = 0)})
  }
  return(wynik)
}


check_ROC("model_neuralnet_PCA", "model_neuralnet_SA") #false
check_ROC("model_randomforest_SA", "model_forward") #true
check_ROC("model_logit_SA", "model_logit_PCA") #true
check_ROC("model_forward", "model_lasso") #0,093
check_ROC("model_forward", "model_logit_full") #0,029
check_ROC("model_randomforest_full", "model_randomforest_SA") #true
check_ROC("model_randomforest_PCA", "model_neuralnet_full") #true



check_LIFT <- function(model1, model2){
  p1 <- shapiro.test(full_eval$LIFT[full_eval$Model == model1])$p.value
  p2 <- shapiro.test(full_eval$LIFT[full_eval$Model == model2])$p.value
  
  if(p1 > 0.05 & p2 > 0.05){
    wynik <-  with(data = full_eval, expr = 
    {t.test(x = LIFT[Model == model1], y = LIFT[Model == model2], 
            alternative = "greater", mu = 0)})
  } else{
    wynik <- with(data = full_eval, expr = 
    {wilcox.test(x = LIFT[Model == model1], y = LIFT[Model == model2], 
                 alternative = "greater", mu = 0)})
  }
  return(wynik)
}


check_LIFT("model_randomforest_SA", "model_randomforest_PCA") #true
check_LIFT("model_lasso", "model_logit_full") #true
check_LIFT("model_logit_full", "model_forward") #true
check_LIFT("model_forward", "model_logit_SA") #true
check_LIFT("model_lasso", "model_forward") #true
check_LIFT("model_lasso", "model_neuralnet_PCA") #true
check_LIFT("model_randomforest_PCA", "model_lasso") #true
check_LIFT("model_logit_SA", "model_logit_PCA") #true


check_ACCURACY <- function(model1, model2){
  p1 <- shapiro.test(full_eval$ACCURACY[full_eval$Model == model1])$p.value
  p2 <- shapiro.test(full_eval$ACCURACY[full_eval$Model == model2])$p.value
  
  if(p1 > 0.05 & p2 > 0.05){
    wynik <-  with(data = full_eval, expr = 
    {t.test(x = ACCURACY[Model == model1], y = ACCURACY[Model == model2], 
            alternative = "greater", mu = 0)})
  } else{
    wynik <- with(data = full_eval, expr = 
    {wilcox.test(x = ACCURACY[Model == model1], y = ACCURACY[Model == model2], 
                 alternative = "greater", mu = 0)})
  }
  return(wynik)
}

check_ACCURACY("model_lasso","model_logit_full") #true
check_ACCURACY("model_lasso","model_forward") #true
check_ACCURACY("model_logit_PCA","model_randomforest_full") #true
check_ACCURACY("model_randomforest_PCA","model_neuralnet_PCA") #true
check_ACCURACY("model_logit_SA","model_logit_PCA") #true
check_ACCURACY("model_randomforest_SA","model_randomforest_PCA") #true
check_ACCURACY("model_neuralnet_SA","model_neuralnet_PCA") #true
