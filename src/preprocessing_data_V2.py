import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data

# Fonctions de preprocessing
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_RainToday

# Fonctions pour compléter les Nas
import complete_nas_functions

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

# A FAIRE preprocessor = Pipeline(['init'])

# STEP 1 preprocess Date
df = preprocess_Date.preprocess_Date(df)

# STEP 2 preprocess RainTomorrow
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)

# STEP 3 remplissage des Nas par les valeurs à proximité #

nas_before_near = pd.DataFrame(df.isna().sum())

# Choix des colonnes à compléter
cols_to_fill = ['MinTemp', 'MaxTemp', 'Rainfall', 
                'Evaporation','Sunshine', 
                'WindGustDir', 'WindGustSpeed', 'WindDir9am', 
                'WindDir3pm','WindSpeed9am', 'WindSpeed3pm', 
                'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Pressure3pm', 
                'Cloud9am', 'Cloud3pm', 
                'Temp9am','Temp3pm']

# création du pipeline de transformers 

pipeline_nearest = \
    complete_nas_functions.create_complete_nas_pipeline(
        cols_to_fill=cols_to_fill,
        preprocess_method="nearest", # choix de la méthode de preprocessing
        distance_max=50) # choix de la distance max où on cherche une station proche

# Fittransform du pipeline (assez long donc je sauvegarde à l'issue
# df_near = complete_nas_transformer_nearest.fit_transform(df)
# df_near.to_csv('../data_saved/df_near.csv')

df_near = pd.read_csv('../data_saved/df_near.csv', 
                      index_col=["id_Location","id_Date"])

nas_after_near = pd.DataFrame(df_near.isna().sum())

# STEP 4 remplissage des Nas restants à par les valeurs médiane par mois et location (méthode Jennifer)
# on applique uniquement aux variables quantitatives

cols_to_fill2 = ['MinTemp', 'MaxTemp', 'Rainfall', 
                'Evaporation','Sunshine', 
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Pressure3pm', 
                'Temp9am','Temp3pm']


pipeline_median_location_month = \
    complete_nas_functions.create_complete_nas_pipeline(
        cols_to_fill=cols_to_fill2,
        preprocess_method="median_location_month")

# fit transform sur l'ensemble du df et sur le df où on a déjà appliqué la fonction nearest
df_median = pipeline_median_location_month.fit_transform(df)
df_near_median = pipeline_median_location_month.fit_transform(df_near)

nas_after_median = pd.DataFrame(df_near_median.isna().sum())

# STEP 5 remplissage des Nas restants pour les variables quantitatives 
# par les modes par location et month

cols_to_fill3 = ["WindGustDir", 'Cloud9am', 'Cloud3pm']
transformer_mode_location_month = \
    complete_nas_functions.create_complete_nas_pipeline(cols_to_fill = cols_to_fill3, 
                                                        preprocess_method= "mode_location_month")

df_near_median_mode = transformer_mode_location_month.fit_transform(df_near_median)

# STEP 6 remplissage des Nas restants pour les variables quantitatives 
# par les modes par location et target (méthode Matthieu)
transformer_mode_target = \
    complete_nas_functions.create_complete_nas_pipeline(
        cols_to_fill = cols_to_fill3, 
        preprocess_method= "mode_location_target")

df_near_median_mode = transformer_mode_target.fit_transform(df_near_median_mode)

nas_after_mode = pd.DataFrame(df_near_median_mode.isna().sum())

print(nas_after_mode)

# STEP 7 preprocess RainToday (en fonction des Rainfall récupérés)
df = preprocess_RainToday.preprocess_RainToday(df_near_median_mode )
df.isna().sum()

# STEP 8 Encodage des variables selon deux types (onehot et trigonométrique (cos et sin))
# On crée des dummies pour ces variables 
vars_onehot = ["Climate", "Year", "Cloud3pm", "Cloud9am"]

# A faire : passer cloud en labelling encoder
# Les variables avec un encodage trigonometrique sont "WindGustDir", "Month" et "Season" 
pipeline_encoding = Pipeline([
    ('Month_encoder', encode_functions.trigo_encoder(col_select="Month")),
    ('Year_encoder', encode_functions.trigo_encoder(col_select="Season")),
    ('WindGustDir_encoder', encode_functions.trigo_encoder(col_select="WindGustDir")),
    ('OneHot_encoder',  encode_functions.encoder_vars(vars_to_encode=vars_onehot))
    ])


df_final = pipeline_encoding.fit_transform(df)
df_final.drop(columns = ["Month", "Season", "WindGustDir"])

df_final.columns
df_final.dtypes
# affichage Nas après preprocessing

nas_after_preprocess = pd.DataFrame(df_final.isna().sum())
nas_after_preprocess = pd.merge(nas_before_preprocess,
                                nas_after_preprocess,
                                left_index=True,
                                right_index=True)

nas_after_preprocess.columns = ['Avant', 'Après']

print("Après Preprocess : \n")
print("Nombre de Nas")
print(nas_after_preprocess)

dim_after_preprocess = df_final.shape
print("Dimensions avant : ", dim_before_preprocess)
print("Dimensions après : ", dim_after_preprocess)

# On retire les colonnes Date et Location qui sont en index
df_final = df_final.drop(columns=["Date", "Location",)


df_final.to_csv('../data_saved/data_preprocessed.csv')


# Scindage du dataset en un échantillon test (20%) et train

feats = df_final.drop(columns="RainTomorrow")
target = df_final["RainTomorrow"]

X_train, X_test, y_train, y_test = \
    train_test_split(feats, target, test_size=0.20, random_state=1234)

# Scaling
# Normalisation (à inclure dans un transformer ou une fonction)

# On fit sur Xtrain complet

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)

print(X_train_scaled.describe())
print(X_test_scaled.describe())

# Sauvegarde fin preprocessing

X_train_scaled.to_csv("../data_saved/X_train_final.csv")
X_test_scaled.to_csv("../data_saved/X_test_final.csv")
y_train.to_csv("../data_saved/y_train_final.csv")
y_test.to_csv("../data_saved/y_test_final.csv")
