# Importer les bibliothèques
import pandas as pd
import numpy as np
import dataframe_image as dfi
from scipy import stats
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("future.no_silent_downcasting", True)
sys.stdout.reconfigure(encoding="utf-8")

# Importer les fonctions 
import init_data
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_Rainfall_RainToday
import preprocess_wind
import preprocess_temperatures
import preprocess_var_quali
import preprocess_pressure_humidity
import encode_functions
import functions_created

# Chargement des données
data_dir = "C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\VSCode Env\\weatherAUS.csv"
df = init_data.initialize_data_weatherAU(data_dir)

# Vérification chargement
print("df head :\n", df.head(), "\n")
print("df describe :\n", df.describe(), "\n")
print("df info :\n")
print(df.info(), "\n")

# Informations sur les NaNs
nas_before_preprocess = pd.DataFrame(df.isna().sum())
print("Avant Preprocess : \n")
print("Nombre de NaNs")
print(nas_before_preprocess)
dim_before_preprocess = df.shape
print("Dimensions : ", dim_before_preprocess, "\n")


########################################################################################################

# Gestion des NAs et preprocessing des variables

# On supprime les colonnes Evaporation, Sunshine et Cloud
df = df.drop(columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"])

# Preprocess de differentes variables
df = preprocess_Date.preprocess_Date(df)
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)
df = preprocess_Rainfall_RainToday.preprocess_Rainfall_RainToday(df)
df = preprocess_wind.pipeline_wind_function(df)
df = preprocess_var_quali.preprocess_median_location_month(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"])

# On retire les derniers Nas
df_final = df.dropna()

# On retire les colonnes Date et Location qui sont en index
df_final = df_final.drop(columns=["Date", "Location"])

# Informations sur les NaNs après preprocessing
nas_after_preprocess = pd.DataFrame(df.isna().sum())
nas_after_preprocess = pd.merge(nas_before_preprocess, nas_after_preprocess,
                                left_index=True, right_index=True)
nas_after_preprocess.columns = ["Avant", "Après"]
print("Après Preprocess : \n")
print("Nombre de Nas")
print(nas_after_preprocess, "\n")
dim_after_preprocess = df.shape
print("Dimensions avant : ", dim_before_preprocess)
print("Dimensions après : ", dim_after_preprocess, "\n")

# Informations sur le dataframe
print("\ndf_final head :\n", df.head(), "\n")
print("\ndf_final info :\n")
print(df_final.info(), "\n\n")


########################################################################################################

## Encodage et séparation des données

# Encodage des dernieres variables catégorielles
df_final = pd.get_dummies(df_final, columns=["Climate", "Season"], drop_first=False)

# Encodage des mois de l'année
month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
                 "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
df_final["Month"] = df_final["Month"].replace(month_mapping)

# Transformation des variables
numeric_cols = df_final.select_dtypes(include=["object", "bool"]).columns
df_final[numeric_cols] = df_final[numeric_cols].astype(int)

# Infos sur le Dataframe
print("\ndf_final info :\n")
print(df_final.info(), "\n\n")

# Sauvegarder df en df_final au format csv
df_final.to_csv("C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\VSCode Env\\df_final.csv")

