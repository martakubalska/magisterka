#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------

source("./load_libraries.R")


#PARAMS
cores <- detectCores() - 1 

#####################

#### READ DATA ####
dane <- data.table::fread("./inputs/train.csv")

###DELETE NAs
dane <- dane %>% select_if(!(colSums(is.na(.))/nrow(.)>=0.1)) ###-10 zmiennych

###REMOVE integer64
dane <- dane %>%
  select(-VAR_0212)

###REMOVE BOOL VARIABLE
dane <- dane %>% select(-VAR_0232)

###NEAR 0 VARANCE
cl <- makeCluster(cores)
registerDoParallel(cl)
nzv_cols <- nearZeroVar(dane, foreach = T, allowParallel = T)
stopCluster(cl)

names(dane)[nzv_cols] -> nzv_vars ###-462 zmienne
dane %>%
  select(-one_of(nzv_vars)) -> dane

###CHECKING WEIRD CHARACTERS
dane %>% select_if(is.character) -> char_cols
unique_char <- sapply(char_cols,unique)


dane %>% select(-VAR_0200, -VAR_0204, -VAR_0075, -VAR_0217) -> dane

###FACTORIZING

#characters - 18 zmiennych
dane <- dane %>% mutate_if(is.character, factor)

#reszta
unique_values <- sapply(dane,unique)

for (i in 1:length(dane)){
  if (!is.factor(dane[,i])){
    if (length(unique_values[[i]])<=10){
      dane[,i] <- factor(make.names(dane[,i]))
    }
  }
}

table(sapply(dane,class))

#ALL AS NUMERIC
dane <- dane %>% mutate_if(is.integer,as.numeric)

table(sapply(dane,class))


####MERGE FACTORS####
factors <- dane %>% select_if(is.factor) %>% select(-target) %>%
  mutate_all(funs(fct_lump(.,n=7))) %>% 
  mutate_all(funs(fct_lump(., prop=0.03)))

factor_cols <- factors %>% names()

dane <- dane %>% 
  select(-one_of(factor_cols)) %>% 
  bind_cols(factors)


##################CREATE TWO DATASETS#####################
set.seed(101) 
sample <- sample.int(n = nrow(dane), size = floor(.7*nrow(dane)), replace = F)
train <- dane[sample, ]
test  <- dane[-sample, ]

test$VAR_0293[which(test$VAR_0293 <0) ] <- 10477

###SKEWNESS
train %>% 
  select(-target) %>% 
  select_if(is.numeric) %>%
  apply(MARGIN = 2, FUN = skewness, na.rm = T) -> skewed

skewed <- which(skewed < -2 | skewed > 2)

train %>% 
  select(-target) %>% 
  select_if(is.numeric) %>% 
  colnames() %>% 
  .[skewed] -> skewed_cols

#negative columns
train %>% 
  select(-target) %>% 
  select_if(is.numeric) %>%
  .[,skewed] %>% 
  apply(MARGIN = 2, FUN = min, na.rm = T) -> col_mins

negative <- which(col_mins<0)
negative_cols <- names(negative)

#logaritm
train %>%
  select(one_of(skewed_cols)) %>%
  select(-one_of(negative_cols)) %>%
  apply(MARGIN = 2, FUN = log1p) %>%
  as.data.frame() -> temp

temp1 <- train %>% select(one_of(negative_cols))

train <- train %>% 
  select(-one_of(skewed_cols)) %>% 
  bind_cols(temp) %>% bind_cols(temp1)

#on the test set
test %>% 
  select(one_of(skewed_cols)) %>%
  select(-one_of(negative_cols)) %>%
  apply(MARGIN = 2, FUN = log1p) %>%
  as.data.frame() -> temp

temp1 <- test %>% select(one_of(negative_cols))  
  
test <- test %>% 
  select(-one_of(skewed_cols)) %>% 
  bind_cols(temp) %>% bind_cols(temp1)

###IMPUTATION
train_num <- train %>% select_if(is.numeric)
numeric_cols <- train %>% select_if(is.numeric) %>% names

cl <- makeCluster(cores)
registerDoParallel(cl)
imp_model <- preProcess(train_num, method = "medianImpute")
train_num <- predict(imp_model, newdata = train_num)
stopCluster(cl)

train <- train %>% select(-one_of(numeric_cols)) %>% bind_cols(train_num)

#on the test set
test_num <- test %>% select_if(is.numeric)

test_num <- predict(imp_model, newdata = test_num)

test <- test %>% select(-one_of(numeric_cols)) %>% bind_cols(test_num)



####CORRELATION####
cl <- makeCluster(cores)
registerDoParallel(cl)
train %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") -> cormat
stopCluster(cl)

findCorrelation(cormat, cutoff = 0.7, names = T) -> cor_vars

corrplot(cormat, method = "color", order="hclust", tl.pos = "n")

train <- train %>% select(-one_of(cor_vars))
test <- test %>% select(-one_of(cor_vars))

####FIND LINEAR COMBINATIONS####
cl<- makeCluster(cores)
registerDoParallel(cl)
lin_comb <- findLinearCombos(as.matrix(train %>% select_if(is.numeric)))
stopCluster(cl)



####STANDARIZING####
std_model <- preProcess(train)

train <- predict(std_model, newdata = train)
test <- predict(std_model, newdata = test)



####PREDICTIVE POWER####
#factors

factors <- train %>% select_if(is.factor)

target_idx <- which(names(factors) %in% "target")

chi_pv <- mapply(function(x, y) chisq.test(x, y)$p.value, factors[, -target_idx], 
                 MoreArgs=list(factors[,target_idx]))

chi_names <- names(which(chi_pv > 0.05))

train <- train %>% select(-chi_names)
test <- test %>% select(-chi_names)

#numeric

numeric <- train %>% select_if(is.numeric) %>% bind_cols(train %>% select(target))

target_idx <- which(names(numeric) %in% "target")

wilcox_pv <- mapply(function(x,y) wilcox.test(x~y)$p.value, numeric[,-target_idx],
       MoreArgs = list(numeric[,target_idx]))

wilcox_names <- names(which(wilcox_pv > 0.05))

train <- train %>% select(-wilcox_names)
test <- test %>% select(-wilcox_names)


####PREDICTIVE POWER - BY LOGITS####
library(ROCR)
library(wrapr)

moc_predykcyjna2 <- function(zmienna){
  
  let(
    list(x = zmienna),
    {
      if(is.factor(train$x) == T){
        model <- glm(target ~ x, 
                     data = train %>% select(target, x), 
                     family = "binomial")
      } else{
        model <- glm(target ~ x + I(x^2), 
                     data = train %>% select(target, x), 
                     family = "binomial")
      }
    }
  )
  
  prog <- predict(model, newdata = train, type = "response")
  pred <- prediction(prog, train$target, label.ordering = levels(train$target))
  auc <- attr(performance(pred,"auc"),"y.values")[[1]]
  
  return(auc)
}

moc_pred <- purrr::map_dbl(.x = names(train)[-1], .f = moc_predykcyjna2)

good_vars <- names(train[,which(moc_pred >0.525)+1])

train <- train %>% select(target, one_of(good_vars))
test <- test %>% select(target, one_of(good_vars))
