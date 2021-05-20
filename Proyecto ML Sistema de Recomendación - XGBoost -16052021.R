#=========================================================================================
# PROYECTO MACHINE LEARNING - SISTEMA DE RECOMENDACIÓN - XGBOOST
# Alvaro Cia, Jose Maria Varela y Roberto Maeso
# Mayo 2021
#========================================================================================

setwd("C:/Users/rmaesobeni001/Desktop/Machine learning/Proyecto")

library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(fasttime)
library(caret)
library(pROC)
library(lubridate)
library(mltools)
library(xgboost)
library(Matrix)
library(Metrics)

#=========================================================================================
#=========================================================================================
# TABLA DE CONTENIDOS
# [1] LOADING DATASET
# [2] DATA CLEANING
# [3] FEATURE ENGINERING 
# [4] DATA PRE-PROCESSING
# [5] MODELLING - XGBOOST
# [6] METRICA- MAP@K 7
# [7] PREDICTIONS NEXT MONTH - MAY 2016
#=========================================================================================
#=========================================================================================

# [1] LOADING DATASET
#=========================================================================================

df<-read.csv('rec_sys_alumnos.csv') #csv original
head(df)
df<- df %>% arrange(mes) %>% as.data.table()
df$month.id<- as.numeric(factor((df$mes)))
df$month.previous.id<- df$month.id - 1

# convertimos a formato fecha
df[,mes:=fastPOSIXct(mes)]
df[,fecha1:=fastPOSIXct(fecha1)]
unique(df$mes)
df$month <- month(df$mes)

sapply(df,function(x)any(is.na(x)))

# cargamos csv con el clustering realizado en otro script - 5 grupos de clientes (finalmente no se ha utilizado)
cl<-read.csv('clustering_5.csv')
head(cl)

# [2] DATA CLEANING
#=========================================================================================
sapply(df, function(x) sum(is.na(x)))

# Nota: El análisis exploratorio de los datos se ha realizado en otro script

# (1) Eliminamos los registros donde para determinados clientes no existe información de varias variables como fecha1, xti_nuevo_cliente,xti_rel,tip_dom,xti_actividad_cliente (se podría sustituir por la moda algunas de estas variables ya que datan de los primeros meses del horizonte temporal para algunos clientes pero los resultados no varían)
# Tipo de domicilio no parece ser útil y fec_ult_cli_1t practicamente todos los datos son mising values 
table(df$tip_dom)
df <- df %>% select(-tip_dom,-fec_ult_cli_1t,-X)
var_na<-c("fecha1","xti_nuevo_cliente","xti_rel","xti_actividad_cliente")
df<- df %>% filter_at(vars(var_na),all_vars(!is.na(.)))

# (2) Mean engagement y xti_rel_1mes parecen tener mismos missing values tambien
table(df$xti_rel_1mes)
# Dado que la casi todos son 1, sustituimos missing values por la categoría más fecuente
df$xti_rel_1mes[is.na(df$xti_rel_1mes)] <- 1
# Para el mean engagement sustituimos por la mediana
df$mean_engagement[is.na(df$mean_engagement)] <- median(df$mean_engagement,na.rm=TRUE)

# (3) Analizamos por separado el resto de variables que cuentan con bastantes missing values: Renta, codidgo provincia, mean engagement 

# Codigo postal solo incluye las provincias de españa. Creamos una categoría aparte con el resto de missing values
unique(df$cod_provincia)
df$cod_provincia[is.na(df$cod_provincia)] <- 0
sum(is.na(df$cod_provincia))

# La renta varía de forma considerable por region, por lo que imputamos los missing values con la mediana de cada provincia (la mayoría de clientes proceden de España)
sum(is.na(df$imp_renta))
new.incomes <-df %>%
  select(cod_provincia) %>%
  merge(df %>%
  group_by(cod_provincia) %>%
  dplyr::summarise(med.income=median(imp_renta)),by="cod_provincia") %>%
  select(cod_provincia,med.income) %>%
  arrange(cod_provincia)
df <- arrange(df,cod_provincia)
df$imp_renta[is.na(df$imp_renta)] <- new.incomes$med.income[is.na(df$imp_renta)]
df$imp_renta[is.na(df$imp_renta)] <- median(df$imp_renta,na.rm=TRUE)
rm(new.incomes)

# (4) Por ultimo, tenemos 3 mising values en dos productos (22 y 23), por lo que asumimos que el cliente no tiene contratados estos productos
df$ind_prod22[is.na(df$ind_prod22)] <- 0
df$ind_prod23[is.na(df$ind_prod23)] <- 0

sapply(df, function(x) sum(is.na(x))) # check non missing values
str(df)

# [3] FEATURE ENGINERING 
#=========================================================================================
features <- names(df)[grepl("ind_",names(df))]
df[,features] <- lapply(df[,features],function(x)as.integer(round(x)))

# LAGS PARA LAS VARIABLES DE INTERES
#-----------------------------------
# creamos una función para seleccionar el número de lags que queremos recoger de cada variable
make_lag <- function(data, # dataframe sobre el que se quieren calcular las lags
                      var.nombre, # variable sobre la que queremos crear la lag
                      lag.m=1,# número de meses (hasta 15 disponibles)
                      by=c("cod_persona","month.id"), # se crean información pasada en función del mes y el cliente
                      na.fill = NA)  
{
  data.sub <- data[,mget(c(by,var.nombre))]
  names(data.sub)[names(data.sub) == var.nombre] <- "original.feature"
  original.month.id <- data.sub$month.id
  added.names <- c()
  for (month.ago in lag.m){
    print(paste("Recogiendo info de",var.nombre,month.ago,"mes(es) atras"))
    colname <- paste("lagged.",var.nombre,".",month.ago,"months.ago",sep="")
    added.names <- c(colname,added.names)
    data.sub <- merge(data.sub,data.sub[,.(cod_persona,month.id=month.ago+original.month.id,lagged.feature=original.feature)],by=by,all.x=TRUE,sort=FALSE)
    names(data.sub)[names(data.sub)=="lagged.feature"] <- colname
  }
  df <- merge(data,data.sub[,c(by,added.names),with=FALSE],by=by,all.x=TRUE,sort=FALSE)
  df[is.na(df)] <- na.fill
  return(df)
}

# Una vez hemos creado la función, pasamos a definir las variables de las que queremos capturar información de periodos pasados
df <- as.data.table(df)
lag_products<-7 # numero de lags para los productos
lag_features<-2 # numero de lags para las variables independendientes
# Ampliamos las variables incluyendo las lags de las features (2 periodos) y los productos (5 periodos)
df <- make_lag(df,'xti_rel',1:lag_features,na.fill=0)
df <- make_lag(df,'mean_engagement',1:lag_features,na.fill=0)
df <- make_lag(df,'xti_actividad_cliente',1:lag_features,na.fill=0)
# Realizamos el mismo proceso para cada producto
df <- make_lag(df,'ind_prod1',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod2',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod3',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod4',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod5',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod6',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod7',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod8',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod9',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod10',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod11',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod12',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod13',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod14',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod15',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod16',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod17',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod18',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod19',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod20',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod21',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod22',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod23',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod24',1:lag_products,na.fill=0)
df <- make_lag(df,'ind_prod25',1:lag_products,na.fill=0)

# PROPOCION DE PRODUCTOS ACTIVOS POR CLIENTE
#-------------------------------------------------------------------
# creamos una variable para cada producto con el % de activación para cada cliente
df<- df %>%
  group_by(cod_persona) %>%
  mutate(prop.ind_prod1 = mean(as.factor(ind_prod1) == 1),
         prop.ind_prod2 = mean(as.factor(ind_prod2) == 1),
         prop.ind_prod3 = mean(as.factor(ind_prod3) == 1),
         prop.ind_prod4 = mean(as.factor(ind_prod4) == 1),
         prop.ind_prod5 = mean(as.factor(ind_prod5) == 1),
         prop.ind_prod6 = mean(as.factor(ind_prod6) == 1),
         prop.ind_prod7 = mean(as.factor(ind_prod7) == 1),
         prop.ind_prod8 = mean(as.factor(ind_prod8) == 1),
         prop.ind_prod9 = mean(as.factor(ind_prod9) == 1),
         prop.ind_prod10 = mean(as.factor(ind_prod10) == 1),
         prop.ind_prod11 = mean(as.factor(ind_prod11) == 1),
         prop.ind_prod12 = mean(as.factor(ind_prod12) == 1),
         prop.ind_prod13 = mean(as.factor(ind_prod13) == 1),
         prop.ind_prod14 = mean(as.factor(ind_prod14) == 1),
         prop.ind_prod15 = mean(as.factor(ind_prod15) == 1),
         prop.ind_prod16 = mean(as.factor(ind_prod16) == 1),
         prop.ind_prod17 = mean(as.factor(ind_prod17) == 1),
         prop.ind_prod18 = mean(as.factor(ind_prod18) == 1),
         prop.ind_prod19 = mean(as.factor(ind_prod19) == 1),
         prop.ind_prod20 = mean(as.factor(ind_prod20) == 1),
         prop.ind_prod21 = mean(as.factor(ind_prod21) == 1),
         prop.ind_prod22 = mean(as.factor(ind_prod22) == 1),
         prop.ind_prod23 = mean(as.factor(ind_prod23) == 1),
         prop.ind_prod24 = mean(as.factor(ind_prod24) == 1),
         prop.ind_prod25 = mean(as.factor(ind_prod25) == 1)
         )

# CLUSTERING - FINALMENTE NO SE USA
#-----------------------------------
# incluimos clusters estimados
# df<-merge(df, cl, by = "cod_persona",all.x = TRUE)
# rm(cl)

# [4] DATA PRE-PROCESSING
#=========================================================================================
# preparamos los datos para la modelización (one-hot encoding y selección de los meses de interes)
levels(df$sexo)<-c(NA,1,0) # 1 para H y 0 para V
df<-na.omit(df)
df$edad<-as.numeric(as.character(df$edad))
df$num_antiguedad<-as.numeric(as.character(df$num_antiguedad))

df$mes <- as.Date(df$mes, format= "%Y-%m-%d")
train.month<-'2016-03-28' #marzo 2016
test.month<-'2016-04-28' #abril 2016

# Filtramos el dataframe para marzo - train- y abril-test-
train<-subset(df, mes==train.month)
test<-subset(df, mes==test.month)

# Guardamos id de los clientes
train.cod_persona<-train$cod_persona
test.cod_persona<-test$cod_persona

# ajustamos test para realizar predicciones
test <- test %>% filter(cod_persona %in% train.cod_persona)
test.cod_persona_ajustado<-test$cod_persona

# eliminamos las variables que ya no necesitamos
train <- train[,-1]
train <- train %>% 
  dplyr::select(-mes,-fecha1,-month.previous.id,-month.id)
test <- test[,-1]
test <- test %>% 
  dplyr::select(-mes,-fecha1,-month.previous.id,-month.id)

# eliminamos también las variables des_canal y pais (TERMINAR DE REVISAR)
train <- train %>% 
  dplyr::select(-pais,-des_canal)
test <- test %>% 
  dplyr::select(-pais,-des_canal)

# one hot enconding all categorical variables c("sexo","xti_empleado","tip_rel_1mes","indresi","indext","xti_extra","id_segmento")
train <- as.data.table(train)
train <- one_hot(train)
test <- as.data.table(test)
test <- one_hot(test)

# Seguimos el mismo proceso que para train y test con los datos clientes abril (total) para predecir mayo
#----------------------------------------------------------------------------------------------------------
# Usamos la base de datos test (abril 2016) para realizar las predicciones para mayo
forecast<-subset(df, mes==test.month)
# Guardamos id de los clientes
forecast.cod_persona<-forecast$cod_persona
# eliminamos las variables que ya no necesitamos
forecast.model <- forecast[,-1]
forecast.model <- forecast.model %>% 
  select(-mes,-fecha1,-month.previous.id,-month.id)
# eliminamos también las variables des_canal y pais
forecast.model <- forecast.model %>% 
  select(-pais,-des_canal)
# one hot enconding all categorical variables c("sexo","xti_empleado","tip_rel_1mes","indresi","indext","xti_extra","id_segmento")
forecast.model <- as.data.table(forecast.model)
forecast.model <- one_hot(forecast.model)


# [5] MODELLING - XGBOOST
#=========================================================================================
# se prepara los conjuntos de datos (entrenamiento, validación y predicción) para la modelización
set.seed(123)

# Entrenamiento - marzo 2016
train.X <- train %>% select(-features)
train.X <- as.matrix(train.X)
train.y <- train %>% select(features) 
train.y <- as.matrix(train.y)

# Validación - abril 2016 (filtrando por clientes que estaban en marzo)
test.X <- test %>% select(-features)
test.X <- as.matrix(test.X)
test.y <- test %>% select(features) 
test.y <- as.matrix(test.y)

# Estimación - abril 2016 (total de clientes)
forecast.X <- forecast.model %>% select(-features)
forecast.X <- as.matrix(forecast.X)
forecast.y <- forecast.model %>% select(features) 
forecast.y <- as.matrix(forecast.y)

results<-list()
results.may<-list()
# creamos un loop para realizar un modelo xgboost binario para cada producto
for(i in 1:ncol(train.y)){
  # convertimos al formato sparse matrix
  dtrain <- xgb.DMatrix(data = train.X, label = train.y[,i])
  dtest <- xgb.DMatrix(data = test.X, label = test.y[,i])
  dmay <- xgb.DMatrix(data = forecast.X, label = forecast.y[,i])
  # modelo xgboost para cada variable
  xgboost <- xgboost(data = dtrain, max.depth = 7, eta = 0.05, nthread = 4,nround = 45, subsample=0.75,
                     objective = "binary:logistic", 
                     verbose =1 ,
                 print_every_n = 1)
  name <- paste('ind_prod',i,sep='_')
  preds<-list(predict(xgboost,dtest))
  preds.may<-list(predict(xgboost,dmay))
  results[[name]] <- preds
  results.may[[name]]<-preds.may
}

# Organizamos predicciones de train y test para obtener MAPK en validación
#--------------------------------------------------------------------------
predictions <- data.frame(matrix(unlist(results), ncol=length(results), byrow=FALSE))
predictions<-cbind(cod_persona=test.cod_persona_ajustado,predictions)
names(predictions) <- gsub(x = names(predictions), pattern = "\\X", replacement = "ind_prod")

predictions$ind_prod1<-0 # el producto 1 no son elegidos para ningun cliente
predictions$ind_prod2<-0 # el producto 2 no son elegidos para ningun cliente

# cambiamos orden columnas
col_order <- c("cod_persona", "ind_prod1", "ind_prod2","ind_prod3","ind_prod4","ind_prod5","ind_prod6","ind_prod7","ind_prod8","ind_prod9","ind_prod10","ind_prod11","ind_prod12","ind_prod13","ind_prod14","ind_prod15","ind_prod16","ind_prod17","ind_prod18", "ind_prod19","ind_prod20", "ind_prod21","ind_prod22","ind_prod23","ind_prod24","ind_prod25")
predictions <- predictions[, col_order]

# restamos predictions sobre dato real para ver productos nuevos contratados
val.y<-predictions[,-1]
val.y <- as.matrix(val.y)
predictions_new<-val.y-train.y
predictions.formated<-data.frame(predictions_new)
predictions.formated<-cbind(cod_persona=test.cod_persona_ajustado,predictions.formated)

# seleccionamos las 7 probabilidades más altas para cada cliente
preds_list <- predictions.formated %>% 
  pivot_longer(cols = -cod_persona) %>%
  group_by(cod_persona) %>%
  top_n(7,value) %>% 
  arrange(cod_persona,desc(value)) %>% 
  select(-value)

# repetimos procedimiento sobre mes actual
test.y <- test %>% select(features)

# productos contratados en el mes de abril (test)
actual<-test.y-train.y
# include cod persona en el test
actual<-data.frame(actual)
actual[actual==-1]<-0 # replace -1 by 0, ya que solo nos interesa productos nuevos contratados
actual<-cbind(cod_persona=test.cod_persona_ajustado, actual)

# escogemos productos contratados por cada cliente
actual <- actual %>% 
  pivot_longer(cols = -cod_persona) %>%
  group_by(cod_persona) %>%
  filter(value==1) %>% select(-value)
cod_persona.contratacion<-unique(actual$cod_persona)

# filtramos por los clientes que verdaderamente contratan
preds_list <- preds_list %>% filter(cod_persona %in% cod_persona.contratacion)

# convertimos a listas pred y test
split_tibble <- function(data, column = 'col') {
  data %>% split(., .[,column]) %>% lapply(., function(x) x[,setdiff(names(x),column)])
}
predic_indprodlist <- split_tibble(as.data.frame(preds_list), 'cod_persona')
actual_indprodlist <- split_tibble(as.data.frame(actual), 'cod_persona')

# # aseguramos que en el orden del cod_persona es el mismo en ambas bases de datos
# predic_indprodlist <- predic_indprodlist[order(sapply(predic_indprodlist, length), decreasing=TRUE)]
# actual_indprodlist <- actual_indprodlist[order(sapply(actual_indprodlist, length), decreasing=TRUE)]

# [6] METRICA- MAP@K 7 (abril 2016, train y test)
#=========================================================================================
print(paste("Validation MAP@7= ",mapk(7,actual_indprodlist,predic_indprodlist)))

# [7] PREDICTIONS NEXT MONTH - MAY 2016
#=========================================================================================
# organizamos las predicciones para mayo realizadas anteriormente
predictions_may <- data.frame(matrix(unlist(results.may), ncol=length(results.may), byrow=FALSE))
predictions_may<-cbind(cod_persona=forecast.cod_persona,predictions_may)
names(predictions_may) <- gsub(x = names(predictions_may), pattern = "\\X", replacement = "ind_prod")
predictions_may$ind_prod1<-0 # el producto 1 no son elegidos para ningun cliente
predictions_may$ind_prod2<-0 # el producto 2 no son elegidos para ningun cliente

# cambiamos orden columnas
predictions_may <- predictions_may[, col_order]

# restamos predictions sobre dato real para ver productos nuevos contratados
forecast.y_ind<-predictions_may[,-1]
forecast.y_ind <- as.matrix(forecast.y_ind)
predictions_new_may<-forecast.y_ind-forecast.y
predictions.formated_may<-data.frame(predictions_new_may)
predictions.formated_may<-cbind(cod_persona=forecast.cod_persona,predictions.formated_may)

# seleccionamos las 7 probabilidades más altas para cada cliente
preds_list_may <- predictions.formated_may %>% 
  pivot_longer(cols = -cod_persona) %>%
  group_by(cod_persona) %>%
  top_n(7,value) %>%
  arrange(cod_persona,desc(value)) %>% 
  select(-value)

# convertimos a formato adecuado para calculo de MAP@K 7
predic_indprodlist_may <- split_tibble(as.data.frame(preds_list_may), 'cod_persona')

# Exportamos predicciones a CSV
#--------------------------------------------------------------------------
forecastproducts <- data.frame(matrix(unlist(predic_indprodlist_may), nrow=length(predic_indprodlist_may), byrow=TRUE))
forecastproducts<-cbind(cod_persona=names(predic_indprodlist_may),forecastproducts)
write.csv(forecastproducts, file="forecast_products.csv",row.names = FALSE)
