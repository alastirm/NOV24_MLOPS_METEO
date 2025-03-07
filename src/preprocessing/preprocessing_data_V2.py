# Importer les bibliothèques
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
#sys.stdout.reconfigure(encoding="utf-8")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data

# Fonctions de preprocessing
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_RainToday
import preprocess_Location
# Fonctions pour compléter les Nas
import complete_nas_functions

# Fonctions encodage
import encode_functions

# pipeline
from sklearn.pipeline import Pipeline

# chargement des données
data_dir = "../../data/weatherAUS.csv"
df = init_data.initialize_data_weatherAU(data_dir)

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

# A FAIRE preprocessor = Pipeline(['init'])

# STEP 1 preprocess Date
df = preprocess_Date.preprocess_Date(df)
df.Year.unique()

# STEP 2 preprocess RainTomorrow
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)

########################################################################################################

# STEP 3 remplissage des Nas par les valeurs à proximité #

nas_before_near = pd.DataFrame(df.isna().sum())

# Choix des colonnes à compléter
cols_to_fill = ['MinTemp', 'MaxTemp', 'Rainfall', 
                'Evaporation', 'Sunshine', 
                'WindGustDir', 'WindGustSpeed', 'WindDir9am', 
                'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 
                'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Pressure3pm', 
                'Cloud9am',  'Cloud3pm', 
                'Temp9am', 'Temp3pm']

# création du pipeline de transformers

pipeline_nearest = \
    complete_nas_functions.create_complete_nas_pipeline(
        cols_to_fill=cols_to_fill,
        preprocess_method="nearest", # choix de la méthode de preprocessing
        distance_max=50) # choix de la distance max où on cherche une station proche

# Fittransform du pipeline (assez long donc je sauvegarde à l'issue)7
# décommenter les deux lignes suivantes au premier run
# df_near = pipeline_nearest.fit_transform(df)
# df_near.to_csv('../../data_saved/df_near.csv')


df_near = pd.read_csv('../../data_saved/df_near.csv', 
                      index_col=["id_Location","id_Date"])

nas_after_near = pd.DataFrame(df_near.isna().sum())

########################################################################################################

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

########################################################################################################*

# STEP 5 remplissage des Nas restants pour les variables quantitatives 
# par les modes par location et month

cols_to_fill3 = ["WindGustDir", 'WindDir9am','WindDir3pm','Cloud9am', 'Cloud3pm']
transformer_mode_location_month = \
    complete_nas_functions.create_complete_nas_pipeline(cols_to_fill = cols_to_fill3, 
                                                        preprocess_method= "mode_location_month")

df_near_median_mode = transformer_mode_location_month.fit_transform(df_near_median)

########################################################################################################

# STEP 6 remplissage des Nas restants pour les variables quantitatives 
# par les modes par location et target (méthode Matthieu)
transformer_mode_target = \
    complete_nas_functions.create_complete_nas_pipeline(
        cols_to_fill = cols_to_fill3, 
        preprocess_method= "mode_location_target")

df_near_median_mode = transformer_mode_target.fit_transform(df_near_median_mode)

nas_after_mode = pd.DataFrame(df_near_median_mode.isna().sum())

print(nas_after_mode)

########################################################################################################

# STEP 7 preprocess RainToday (en fonction des Rainfall récupérés)
df = preprocess_RainToday.preprocess_RainToday(df_near_median_mode)
df.isna().sum()


# STEP 8 preprocess Location (Méthode Mathieu)
df = preprocess_Location.preprocess_Location(df_near_median_mode)
df.isna().sum()

########################################################################################################

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

df.Year.unique()

########################################################################################################

# STEP 9 Encodage des variables selon deux types (onehot et trigonométrique)
# On crée des dummies pour ces variables 
vars_onehot = ["Climate", "Year", "Location"]

# On crée des labels de 1 à n-1 classes pour ces colonnes
vars_labels = ['Cloud9am', 'Cloud3pm']
# A faire : passer cloud en labelling encoder

# Les variables avec un encodage trigonometrique sont "WindGustDir", "Month" et "Season" 
pipeline_encoding = Pipeline([
    ('Month_encoder', encode_functions.trigo_encoder(col_select="Month")),
    ('Year_encoder', encode_functions.trigo_encoder(col_select="Season")),
    ('WindGustDir_encoder', encode_functions.trigo_encoder(col_select="WindGustDir")),
    ('WindDir9am_encoder', encode_functions.trigo_encoder(col_select="WindDir9am")),
    ('WindDir3pm_encoder', encode_functions.trigo_encoder(col_select="WindDir3pm")),
    ('OneHot_encoder',  encode_functions.encoder_vars(
        vars_to_encode=vars_onehot, 
        encoder="OneHotEncoder"))
    ])

df_final = pipeline_encoding.fit_transform(df)

# drop les colonnes encodées en nouvelles colonnes
df_final = df_final.drop(columns=["Month", "Season", "WindGustDir", "WindDir9am", "WindDir3pm"])

########################################################################################################

# Sauvegarde du dataset complet 
df_final.to_csv('../../data_saved/data_preprocessed_V2.csv')

df_final.columns

########################################################################################################

# STEP 10 Séparation des données par Location

# Sauvegarder la liste de toutes les Location
locations = np.unique(df_final.index.get_level_values(0).values)
locations_dummies = "Location_" + locations
df_final["Location"] = df_final.index.get_level_values(0).values
df_final = df_final.drop(columns = locations_dummies)

base_dir = Path(__file__).resolve().parent
output_location = base_dir / "../../data_saved/data_location_V2"
if not output_location.exists():
    output_location.mkdir(parents=True)
groupes = df_final.groupby("Location")
for location, groupe in groupes:
    output_path = os.path.join(output_location, f"df_{location}.csv")
    groupe.to_csv(output_path, index=True)
    print(f"Le Dataframe df_{location} a été créé")


# Nettoyage des fichiers créés
def remove_columns_NAN(location_names, base_dir:Path, threshold:float=0.3):
    for location_name in location_names:
        df_location_path = base_dir / "../../data_saved/data_location_V2" / f"df_{location_name}.csv"
        if not df_location_path.exists():
            print(f"!! Fichier pour '{location_name}' non trouvé !!")
            continue

        # Chargement des données
        df_location = pd.read_csv(df_location_path)

        # Retrait des colonnes inutiles spécifiques à une location
        df_location = df_location.drop(
            columns=["Date", "Location", 
                     'Climate_Desert', 'Climate_Grassland',
                     'Climate_Subtropical', 'Climate_Temperate', 
                     'Climate_Tropical',
                     'sin_lon','cos_lon', 'sin_lat','cos_lat'])

        

        # Calcul du pourcentage de valeurs manquantes
        missing_percentages = df_location.isna().mean()

        # Colonnes à conserver
        columns_to_keep = missing_percentages[missing_percentages <= threshold].index
        df_location_cleaned = df_location[columns_to_keep]

        # Enlever les dernières NaN
        df_location_cleaned = df_location_cleaned.dropna()

        # Enregistrement
        output_path = base_dir / "../../data_saved/data_location_V2" / f"df_{location_name}.csv"
        df_location_cleaned.to_csv(output_path, index=False)
        print(f"Le Fichier {location_name} a été nettoyé et enregistré")

# Appliquer la fonction aux df_location
remove_columns_NAN(location_names=locations, base_dir=base_dir, threshold=0.3)

# Test affichage du pipeline global : 

preprocessor = Pipeline(
    steps = [
        ('Nearest', pipeline_nearest),
        ('Median_location_month', pipeline_median_location_month),
        ('Mode_location_month', transformer_mode_location_month),
        ('Mode_location_target', transformer_mode_target),
        ('Encoding', pipeline_encoding)
    ]
)

preprocessor