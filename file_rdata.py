import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# Abilita la conversione automatica da R a pandas DataFrame
pandas2ri.activate()

# Carica il file .rdata
ro.r['load']('path_al_tuo_file.rdata')

# Trova il nome dell'oggetto in R che vuoi convertire
# Stampa tutti gli oggetti disponibili nel workspace R
print(ro.r['ls']())

# Sostituisci 'nome_dataset_R' con il nome effettivo del tuo dataset in R
dataframe = pandas2ri.ri2py(ro.r['nome_dataset_R'])

# Ora 'dataframe' è un DataFrame pandas
print(dataframe.head())


