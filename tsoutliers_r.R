library(tsoutliers)
library(forecast)

# Carica i dati
data <- read.csv("./data/Dataset/Consumer_power.csv")

# Controlla la struttura e i primi valori del dataset
head(data)
str(data)

# Applica tsoutliers alle colonne da Cons.1 a Cons.7
for(i in 1:14) {
  col_name <- paste("Cons.", i, sep = "")  # Crea il nome della colonna corretto
  
  # Converti la colonna in un oggetto time series con una frequenza di 96 (intervalli di 15 minuti)
  cons_ts <- ts(data[[col_name]], frequency = 96)
  
  # Applica tsoutliers
  outliers_result <- tsoutliers(cons_ts)
  
  # Estrai solo i valori corretti (aggiustati per gli outlier)
  adjusted_values <- outliers_result$yadj
  
  # Crea un indice per gli elementi sostituiti
  replacements <- !is.na(adjusted_values) & (data[[col_name]] != adjusted_values)
  
  # Sostituisci solo i valori dove sono stati rilevati outlier
  data[replacements, col_name] <- adjusted_values[replacements]
}

# Salva i dati aggiornati in un nuovo file CSV
write.csv(data, "./data/Dataset/Updated_Consumer_power.csv", row.names = FALSE)
