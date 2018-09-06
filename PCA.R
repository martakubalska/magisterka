#--------------------------------------------------------------------------------
#                       MARTA KUBALSKA - PRACA MAGISTERSKA
#--------------------------------------------------------------------------------

source("./load_libraries.R")

#---------------------------------ANALIZA GLOWNYCH SKLADOWYCH-------------------#
set.seed(1994)

train_num <- train %>% select_if(is.numeric)
train_fac <- train %>% select(-target) %>% select_if(is.factor)

pcamix_train <- PCAmix(X.quanti = as.matrix(train_num),
                       X.quali = train_fac,
                       rename.level = T,
                       ndim = 150)

opis_pca <- data.frame(pcamix_train$eig)
opis_pca$pca <- stringr::str_replace(rownames(pcamix_train$eig), pattern = "dim ", replacement = "")



###103 PC potrzebne do wyjasnienia 70% zmiennosci

train_pca <- predict(pcamix_train, 
                     X.quanti = as.matrix(train %>% 
                                            select_if(is.numeric)), 
                     X.quali = as.matrix(train %>%
                                           select_if(is.factor) %>%
                                           select(-target)), 
                     rename.level = T)

train_pca <- data.frame(train_pca)
train_pca <- bind_cols(select(train, target), train_pca)

test_pca <- predict(pcamix_train, 
                     X.quanti = as.matrix(test %>% 
                                            select_if(is.numeric)), 
                     X.quali = as.matrix(test %>%
                                           select_if(is.factor) %>%
                                           select(-target)), 
                     rename.level = T)

test_pca <- data.frame(test_pca)
test_pca <- bind_cols(select(test, target), test_pca)


####WIZUALIZACJE####
kolor <- RColorBrewer::brewer.pal(n = 5, name = "Set3")[5]
kolor2 <- RColorBrewer::brewer.pal(n = 3, name = "Set1")[2]

#procentt objasnianej zmiennosci
q <- ggplot(opis_pca[1:80,], aes(x = forcats::fct_reorder(forcats::as_factor(pca), Cumulative), 
                                  y = Proportion))
q + geom_point(alpha = 1, color = kolor2, cex =2) +
  labs(
    title = "Odestek zmiennosci objasniany przez kolejne skladowe",
    subtitle = "Pierwszych 80 skladowych",
    x = "Glowne skladowe",
    y = "Odsetek objasnianej zmiennosci") +
  theme_bw()

#wartosci wlasne

m <- ggplot(opis_pca[1:200,], aes(x = forcats::fct_reorder(forcats::as_factor(pca), Cumulative), 
                                 y = Eigenvalue))
m + geom_point(alpha = 1, color = kolor2, cex =1) +
  theme(axis.text.x=element_blank()) +
  geom_vline(aes(xintercept = 159), color = "dark green", size = 1, linetype = 1) +
  geom_hline(aes(yintercept = 1), color = "red", size = 1, linetype = 1) +
  labs(
    title = "Wartosci wlasne",
    x = "Glowne skladowe",
    y = "Wartosci wlasne") +
  annotate(geom = "text", x = 25, y = -1, label = "Wartosc wlasna rowna 1", color = "red") +
  annotate(geom = "text", x = 145, y = 75, label = "159 skladowa", color = "dark green") 
  theme_bw()


#skumulowany procent objasnianej zmiennosci
p <- ggplot(opis_pca[1:120,], 
            aes(x = forcats::fct_reorder(forcats::as_factor(pca), Cumulative), y = Cumulative)) + 
  geom_col(fill = kolor) + 
  geom_vline(aes(xintercept = 15), color = "green", size = 1, linetype = 1) +
  geom_hline(aes(yintercept = 70), color = "red", size = 1, linetype = 1) + 
  labs(x = "Glowne skladowe",
       y = "Skumulowany % odwzorowanej wariancji",
       title = "Skumulowany procent odwzorowanej wariancji dla kolejnych glownych skladowych") +
  annotate(geom = "text", x = 8, y = 75, label = "70% wariancji", color = "red") + 
  theme_bw()

p

#wizualizacja pierwszych 2 glownych skladowych
set.seed(1994)
probka <- runif(nrow(train_pca))
p <- ggplot(train_pca[probka <= 0.5,], aes(x = dim1, y = dim2, fill = target, col = target))
p + geom_point(alpha = 0.2) +
  labs(
    title = "Zmienna objasniana vs dwie pierwsze skladowe",
    subtitle = "Probka 50% obserwacji",
    x = "Pierwsza glowna skladowa",
    y = "Druga glowna skladowa") +
  theme_bw()


####STANDARIZING####
std_model_PCA <- preProcess(train_pca)

train_pca <- predict(std_model_PCA, newdata = train_pca)
test_pca <- predict(std_model_PCA, newdata = test_pca)


