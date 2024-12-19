import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data


# chargement des données
data_dir = "./Datasets/archive/weatherAUS_091224.csv"
df = init_data.initialize_data_weatherAU(data_dir)

# Vérification chargement

df.head()
df.describe()

# Scindage du dataset en un échantillon test (20%) et train
feats = df.drop(columns = "RainTomorrow")
target = df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=1234)

# gestion des NAs et preprocessing
print(df[["Date"]].head())

# Exemple en important les fonctions de preprocess_Date.py 
import preprocess_Date 

X_train = preprocess_Date.preprocess_Date(X_train)
X_test = preprocess_Date.preprocess_Date(X_test)

print(X_train[["Date","Month","Year"]].head())
print(X_test[["Date","Month","Year"]].head())

