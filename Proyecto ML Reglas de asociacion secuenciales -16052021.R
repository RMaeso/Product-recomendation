#=========================================================================================
# PROYECTO MACHINE LEARNING - SEQUENCIAL ASSOCIATION RULES
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
library(arulesSequences)

#=========================================================================================
#=========================================================================================
# TABLA DE CONTENIDOS
# [1] LOADING DATASET
# [2] DATA CLEANING
# [3] FEATURE ENGINERING 
# [4] MODELING - SPADE
# [5] CLEAN RESULTS FOR INTERPRETATION AND EXPORT TABLE TO CSV
#=========================================================================================
#=========================================================================================

# Nota: Con este modelo no se realizan predicciones, solo el análisis exploratorio de los productos
# para identificar patterns a nivel agregado de clientes 

# [1] LOADING DATASET
#=========================================================================================
df<-read.csv('rec_sys_alumnos.csv')
head(df)
df <- df %>% arrange(mes) %>% as.data.table()
df$month.id <- as.numeric(factor((df$mes)))

# convertimos a formato fecha
df[,mes:=fastPOSIXct(mes)]
unique(df$mes)

sapply(df,function(x)any(is.na(x)))

# [2] DATA CLEANING
#=========================================================================================
# seleccionamos las columnas target
products <- names(df)[grepl("ind_",names(df))]
# Para aplicar las reglas de asociación limpiamos la base de datos para quedarnos unicamente 
# con los productos, id del cliente y la fecha actual
df<-df %>% select(cod_persona,mes,products)

sapply(df, function(x) sum(is.na(x)))

# (1) Tenemos 31 mising values en dos productos (22 y 23), por lo que asumimos que el cliente no tiene contratados estos productos
df$ind_prod22[is.na(df$ind_prod22)] <- 0
df$ind_prod23[is.na(df$ind_prod23)] <- 0
df$ind_prod22<-as.integer(df$ind_prod22)
df$ind_prod23<-as.integer(df$ind_prod23)

sapply(df, function(x) sum(is.na(x))) # check non missing values
str(df)

# [3] FEATURE ENGINERING 
#=========================================================================================
# Transformamos los datos a una matriz de transacción para poder aplicar posteriormente sequence association rules
df.format <-df %>%  
  pivot_longer(cols=starts_with("ind_prod"), names_to = "ind_prod")

# Mes inicial de los datos
start_month <- "2015-01-28"

# Guardamos los codigos de los clientes
cod_personas<-df.format$cod_persona

# Reemplazamos cod_persona por un ID de 1 a X clientes
df.format$cod_persona<-as.character(df.format$cod_persona)
df.format <- df.format %>%
  mutate(id = group_indices(., factor(cod_persona, levels = unique(cod_persona))))
df.format <- df.format %>% select(-cod_persona) %>% 
  select(id, everything())
# Filtramos cuando value==1
df.format <- df.format %>% filter(value==1) %>% select(-value)

# Preparamos dataframe por id y mes
trans_sequence <- df.format %>%
  group_by(id, mes) %>%
  summarize(SIZE = n(),
    ind_prod = paste(as.character(ind_prod), collapse = ';'))

# Convertimos el ID (cod_persona) y evento (mes) a factores
sequence_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}
trans_sequence$eventID <- sequence_months(trans_sequence$mes, start_month)
trans_sequence = trans_sequence[,c(1,5,3,4)]
names(trans_sequence) = c("sequenceID", "eventID", "SIZE", "ind_prod")
trans_sequence <- data.frame(lapply(trans_sequence, as.factor))
trans_sequence<-trans_sequence %>% arrange(sequenceID,eventID)
trans_sequence<-trans_sequence %>% filter(eventID!=0) # eliminamos eventID 0, primer mes de la base de datos

# Convertimos a un matriz transaccional para posteriormente aplicar la libreria (la forma más sencilla de realizar este proceso es
# convirtiendo el dataset a txt y posteriormente leyendolo con la funcion read_baskets)
write.table(trans_sequence, "sequence.txt", sep=";", row.names = FALSE, col.names = FALSE, quote = FALSE)
trans_matrix <- read_baskets("sequence.txt", sep=";" ,info = c("sequenceID","eventID","SIZE"))

# [4] MODELING - Sequential Pattern Discovery using Equivalence classes
#------------------------------------------------------------------
# Obtenemos los productos más frecuentes activados y sus correspondientes support values
seq.ass.rule <- cspade(trans_matrix, parameter = list(support = 0.1), control = list(verbose = TRUE))
seq.ass.rule.df <- as(seq.ass.rule, "data.frame")
summary(seq.ass.rule) # con este función podemos obtener las principales estadisticas (frequent items, )

# Obtenemos las reglas temporales para los productos activados más frecuentemente
freq.ass.rule <- as(ruleInduction(seq.ass.rule, confidence = 0.5, control = list(verbose = TRUE)), "data.frame")
freq.ass.rule

# [5] CLEAN RESULTS FOR INTERPRETATION AND EXPORT TABLE TO CSV
#------------------------------------------------------------------

# Analizamos y separamos los productos más comunes que tiene los clientes
freq.ass.rule$rulecount <- as.character(freq.ass.rule$rule)
max_col <- max(sapply(strsplit(freq.ass.rule$rulecount,' => '),length))
r_sep <- separate(data = freq.ass.rule, col = rule, into = paste0("Time",1:max_col), sep = " => ")
r_sep$Time2 <- substring(r_sep$Time2,3,nchar(r_sep$Time2)-2)

# Realizamos lo mismo pero con las predicciones (es decir a partir de los productos que tienen los clientes cuales son los más probables de contratar en el mes siguiente)
max_time1 <- max(sapply(strsplit(r_sep$Time1,'},'),length))
r_sep$TimeClean <- substring(r_sep$Time1,3,nchar(r_sep$Time1)-2)
r_sep$TimeClean <- gsub("\\},\\{", "zzz", r_sep$TimeClean)
r_sep_items <- separate(data = r_sep, col = TimeClean, into = paste0("Previous_Items",1:max_time1), sep = "zzz")
r_sep_items

# Limpiamos las reglas para construir un dataframe de izquierda a derecha, donde aparezcan productos actuales y recomendados en el mes siguiente
r_shift_na <- r_sep_items

for (i in seq(1, nrow(r_shift_na))){
  for (col in seq(8, (6+max_time1))){
    if (is.na(r_shift_na[i,col])==TRUE){
      r_shift_na[i,col] <- r_shift_na[i,col-1]
      r_shift_na[i,col-1] <- NA  
    }
  }
}
names(r_shift_na)[2] <- "Predicted_Items"

# construimos la base de datos final, donde aparece el producto(s) y los productos recomendados a partir de las reglas de asociacion
cols <- c(7:(6+max_time1), 2:5)
temporal_rules <- r_shift_na[,cols]
temporal_rules <- temporal_rules[order(-temporal_rules$lift, -temporal_rules$confidence, 
                                       -temporal_rules$support, temporal_rules$Predicted_Items),]

write.csv(as.data.frame(temporal_rules), file = "TemporalRules.csv", row.names = FALSE, na="")



