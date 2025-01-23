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
import preprocess_var_quali
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
df = preprocess_var_quali.preprocess_median_location_month(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"])

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

# sauvegarde du dataset complet (libre à vous de sauvegarder avant)
df_final.to_csv('../data_saved/df_final.csv')
