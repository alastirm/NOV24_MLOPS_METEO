import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

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

# Autres fonctions ajoutées
import functions_created

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
df.isna().sum()

# preprocess Rainfall et RainToday
df = preprocess_Rainfall_RainToday.preprocess_Rainfall_RainToday(df)
df.isna().sum()

# preprocess wind
df = preprocess_wind.pipeline_wind_function(df)
df.isna().sum()

# preprocess temperatures
df = preprocess_temperatures.preprocess_temperature_mean(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"])
df.isna().sum()


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
df.isna().sum()

# dist_mat = functions_created.create_distance_matrix()
# df_test = functions_created.complete_na_neareststation(df, variable="MinTemp", distance_max=50)

# affichage Nas après preprocessing

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

df_final.to_csv('../data_saved/data_mat.csv')
print(df_final.columns)
print('data_mat saved')

# Encodage des variables selon deux types (onehot et trigonométrique (cos et sin))
# On crée des dummies pour ces variables
vars_onehot = ["Climate", "Year"]
# Les variables avec un encodage trigonometrique sont "Month" et "Season"
pipeline_encoding = Pipeline([
    ('Month_encoder', encode_functions.trigo_encoder(col_select="Month")),
    ('Year_encoder', encode_functions.trigo_encoder(col_select="Season")),
    ('OneHot_encoder',  encode_functions.encoder_vars(vars_to_encode=vars_onehot))
    ])

df_final = pipeline_encoding.fit_transform(df_final)

df_final.iloc[:,-20:-1].describe()

# On retire les colonnes Date et Location qui sont en index
df_final = df_final.drop(columns=["Date", "Location", "Month", "Season"])
print(df_final.isna().sum())
print(df_final.dtypes)

df_final.to_csv('../data_saved/data_preprocessed.csv')

# On sélectionne les features à garder

feats_selected = ['Year', 'Month', 'Season',
                  'RainToday', 'Rainfall',
                  'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm',
                  'Humidity9am', 'Humidity3pm',
                  'Pressure9am', 'Pressure3pm',
                  'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']

# Scindage du dataset en un échantillon test (20%) et train

feats = df_final.drop(columns="RainTomorrow")
target = df_final["RainTomorrow"]

X_train, X_test, y_train, y_test = \
    train_test_split(feats, target, test_size=0.20, random_state=1234)

# Scaling
# Normalisation (à inclure dans un transformer ou une fonction)
# Choix des colonnes à scaler avec le MinMaxScaler
vars_to_scale = ['Rainfall',
                 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm',
                 'Humidity9am', 'Humidity3pm',
                 'Pressure9am', 'Pressure3pm',
                 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']

# On fit sur Xtrain
scaler = MinMaxScaler().fit(X_train[vars_to_scale])

X_train_scaled = X_train
X_train_scaled[vars_to_scale] = scaler.transform(X_train[vars_to_scale])
X_test_scaled = X_test
X_test_scaled[vars_to_scale] = scaler.transform(X_test[vars_to_scale])

print(X_train_scaled[vars_to_scale].describe())
print(X_test_scaled[vars_to_scale].describe())

# Standardisation (à inclure dans un transformer ou une fonction)
# Choix des colonnes à scaler avec le StandardScaler
# vars_to_scale2 = []

# scaler2 = StandardScaler().fit(X_train[vars_to_scale2])

# X_train_scaled[vars_to_scale2] = scaler2.transform(X_train_scaled[vars_to_scale2])
# X_test_scaled[vars_to_scale2] = scaler2.transform(X_test_scaled[vars_to_scale2])

# print(X_train_scaled[vars_to_scale2].describe())
# print(X_test_scaled[vars_to_scale2].describe())

# Sauvegarde fin preprocessing
X_train_scaled.to_csv("../data_saved/X_train_final.csv")
X_test_scaled.to_csv("../data_saved/X_test_final.csv")
y_train.to_csv("../data_saved/y_train_final.csv")
y_test.to_csv("../data_saved/y_test_final.csv")
