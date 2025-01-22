import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data

# Fonctions de preprocessing
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_Rainfall_RainToday
import preprocess_wind
import preprocess_temperatures
import preprocess_pressure_humidity

# Fonctions encodage
import encode_functions

# pipeline
from sklearn.pipeline import Pipeline

# chargement des données
data_dir = "../data/weatherAUS.csv"
df = init_data.initialize_data_weatherAU(data_dir)

# Vérification chargement
df.head()
df.describe()

# print informations

nas_before_preprocess = pd.DataFrame(df.isna().sum())
print("Avant Preprocess : \n")
print("Nombre de Nas")
print(nas_before_preprocess)

dim_before_preprocess = df.shape
print("Dimensions : ", dim_before_preprocess)

# gestion des NAs et preprocessing des variables

# On supprime les colonnes Evaporation, Sunshine et Cloud
df = df.drop(columns=["Evaporation", 'Sunshine', 'Cloud9am', 'Cloud3pm'])

# preprocess Date
df = preprocess_Date.preprocess_Date(df)

# preprocess Variable cible
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)

# preprocess Rainfall et RainToday
df = preprocess_Rainfall_RainToday.preprocess_Rainfall_RainToday(df)

# preprocess wind
df = preprocess_wind.pipeline_wind_function(df)

# preprocess temperatures
df = preprocess_temperatures.preprocess_temperature_mean(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"])

# preprocess pressures et humidities
# J'ai du ajouter un réindexage car je ne sais pas pourquoi mais la
# fonction remplissage_na_voisinnage modifie les index et ajoute Location en index

columns=["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"]
for col in columns:
    df = df.groupby("Location").apply(preprocess_pressure_humidity.remplissage_na_voisinnage, column=col, global_data=df[col])
    print(df.index)
    df = df.set_index(["Location", "Date"], drop=False)
    df = df.reindex(df.index.rename({"Location" : "id_Location", "Date" : "id_Date"}))

# df = preprocess_pressure_humidity.remplir_na(df, columns=["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"])

# Encodage des variables selon deux types (onehot et trigonométrique (cos et sin))

# On crée des dummies pour ces variables
vars_onehot = ["Climate", "Year"]
# Les variables avec un encodage trigonometrique sont "Month" et "Season"
pipeline_encoding = Pipeline([
    ('Month_encoder', encode_functions.trigo_encoder(col_select="Month")),
    ('Year_encoder', encode_functions.trigo_encoder(col_select="Season")),
    ('OneHot_encoder',  encode_functions.encoder_vars(vars_to_encode=vars_onehot))
    ])

df = pipeline_encoding.fit_transform(df)

# On retire les colonnes Date et Location qui sont en index
# Month et Season qui sont encodées 
df = df.drop(columns=["Date", "Location", "Month", "Season"])

# Nas after preprocess
nas_after_preprocess = pd.DataFrame(df.isna().sum())
nas_after_preprocess = pd.merge(nas_before_preprocess,
                                nas_after_preprocess,
                                left_index=True,
                                right_index=True)

nas_after_preprocess.columns = ['Avant', 'Après']

print("Après Preprocess : \n")
print("Nombre de Nas")
print(nas_after_preprocess)

dim_after_preprocess = df.shape
print("Dimensions avant : ", dim_before_preprocess)
print("Dimensions après : ", dim_after_preprocess)

# On retire les derniers Nas (à faire après avoir géré toutes les colonnes)
df_final = df.dropna()
df_final.columns

# sauvegarde (libre à vous de sauvegarder avant)
df_final.to_csv('../data_saved/data_preprocessed.csv')


# # Scindage du dataset en un échantillon test (20%) et train
# feats = df_final.drop(columns="RainTomorrow")
# target = df_final["RainTomorrow"]

# X_train, X_test, y_train, y_test = \
#     train_test_split(feats, target, test_size=0.20, random_state=1234)

# # Scaling
# # Normalisation (à inclure dans un transformer ou une fonction)

# vars_to_scale = 'all'

# # On fit sur Xtrain complet
# scaler = MinMaxScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)
    
# # Sauvegarde fin preprocessing
# X_train_scaled.to_csv("../data_saved/X_train_final.csv")
# X_test_scaled.to_csv("../data_saved/X_test_final.csv")
# y_train.to_csv("../data_saved/y_train_final.csv")
# y_test.to_csv("../data_saved/y_test_final.csv")
