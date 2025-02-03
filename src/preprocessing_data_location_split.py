# Importer les bibliothèques
from pathlib import Path
import pandas as pd
import numpy as np
import dataframe_image as dfi
from scipy import stats
import os
import sys
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
data_dir = "weatherAUS.csv"
df = init_data.initialize_data_weatherAU(data_dir)

# Sauvegarder la liste de toutes les Location
locations = df["Location"].unique()

# Vérification chargement
print("\n\n")
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

## Gestion des NAs et preprocessing des variables

# Preprocess de differentes variables
df = preprocess_Date.preprocess_Date(df)
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)
df = preprocess_Rainfall_RainToday.preprocess_Rainfall_RainToday(df)
df = preprocess_wind.pipeline_wind_function(df)
df = preprocess_var_quali.preprocess_median_location_month(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm", 
                                                                        "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"])
# NAN dans Evaporation, Sunshine, Cloud9am, et Cloud3pm traitées après la séparation par Location

# Encodage des variables catégorielles
df = pd.get_dummies(df, columns=["Season"], drop_first=False)
month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
df["Month"] = df["Month"].replace(month_mapping)

# Séparation des données par Location
base_dir = Path(__file__).resolve().parent
output_location = base_dir / "data_location"
if not output_location.exists():
    output_location.mkdir(parents=True)
groupes = df.groupby("Location")
for location, groupe in groupes:
    output_path = os.path.join(output_location, f"df_{location}.csv")
    groupe.to_csv(output_path, index=True)
    print(f"Le Dataframe df_{location} a été créé")

# Nettoyage des fichiers créés
def remove_columns_NAN(location_names, base_dir:Path, threshold:float=0.3):
    for location_name in location_names:
        df_location_path = base_dir / "data_location" / f"df_{location_name}.csv"
        if not df_location_path.exists():
            print(f"!! Fichier pour '{location_name}' non trouvé !!")
            continue

        # Chargement des données
        df_location = pd.read_csv(df_location_path)

        # Retrait des colonnes inutiles
        df_location = df_location.drop(columns=["Date", "Location", "Climate"])

        # Calcul du pourcentage de valeurs manquantes
        missing_percentages = df_location.isna().mean()

        # Colonnes à conserver
        columns_to_keep = missing_percentages[missing_percentages <= threshold].index
        df_location_cleaned = df_location[columns_to_keep]

        # Enlever les dernières NaN
        df_location_cleaned = df_location_cleaned.dropna()

        # Enregistrement
        output_path = base_dir / "data_location" / f"df_{location_name}.csv"
        df_location_cleaned.to_csv(output_path, index=False)
        print(f"Le Fichier {location_name} a été nettoyé et enregistré")

# Appliquer la fonction aux df_location
remove_columns_NAN(location_names=locations, base_dir=base_dir, threshold=0.3)

