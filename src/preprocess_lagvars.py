# Importer les bibliothèques
from pathlib import Path
import pandas as pd
import numpy as np
import dataframe_image as dfi
from scipy import stats
import os
import sys
#sys.stdout.reconfigure(encoding="utf-8")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

import init_data
import preprocess_Date

# Fonctions encodage
import encode_functions

# pipeline
from sklearn.pipeline import Pipeline

import functions_created

def add_lagdelta_vars(df, lag_vars, diff_vars):

    # création listes de dates
    data_dir = "../data/weatherAUS.csv"
    df_init = init_data.initialize_data_weatherAU(data_dir)
    df_init = preprocess_Date.preprocess_Date(df_init)
    df_dates, date_list = functions_created.create_date_list(df_init)

    # liste des locations dans le df
    location_list = df.index.get_level_values(0).unique()

    # création d'un df avec toutes les dates 
    # pour les problèmes de rupture de séries

    for select_location in location_list:
    
        df_location = df_dates
        df_location.dtypes
        df.index.dtypes
        df_location["Location"] = select_location
        df_location = df_location.set_index(["Location", "Dates"])
        df_location = df_location.reindex(df_location.index.rename({"Location": "id_Location", "Dates" : "id_Date"}))

        df_location = df_location.merge(df,
                            left_index=True , right_index=True, 
                            how="left")
        
        if select_location == location_list[0]:
            df_alldates = df_location
        else:
            df_alldates = pd.concat([df_alldates, df_location])

    df_alldates.isna().sum()

    # variables pour lesquelles on calcule 3 lags de 3 jours et une moyenne sur 3 jours

    for variable in lag_vars :

        df_alldates[variable + "_lag1"] = df_alldates.groupby("id_Location")[variable].shift(periods = 1)
        df_alldates[variable + "_lag2"] = df_alldates.groupby("id_Location")[variable].shift(periods = 2)
        df_alldates[variable + "_lag3"] = df_alldates.groupby("id_Location")[variable].shift(periods = 3)
        df_alldates[variable + "_mean3j"] = round((df_alldates[variable + "_lag1"] + df_alldates[variable + "_lag2"] + df_alldates[variable + "_lag3"])/3, ndigits=10)
        
        # supprime les valeurs où la variable initiale est Na
        df_alldates.loc[df_alldates[variable].isna(), variable + "_lag1"] = np.nan
        df_alldates.loc[df_alldates[variable].isna(), variable + "_lag2"] = np.nan
        df_alldates.loc[df_alldates[variable].isna(), variable + "_lag3"] = np.nan
        df_alldates.loc[df_alldates[variable].isna(), variable + "_mean3j"] = np.nan


    # variables pour lesquelles on calcule la variation entre 9am et 3pm

    for variable in diff_vars :
        df_alldates[variable + "_delta"] = \
            round((df_alldates[variable + "3pm"] - df_alldates[variable + "9am"])/df_alldates[variable + "9am"]*100,ndigits=3)

        # supprime les valeurs où la variable initiale est Na
        df_alldates.loc[df_alldates[variable + "3pm"].isna(), variable + "_delta"] = np.nan
        df_alldates.loc[df_alldates[variable + "9am"].isna(), variable + "_delta"] = np.nan

    # suppression des lignes complètement vides
    df_alldates = df_alldates.drop(columns="Year")
    df_return = df_alldates.dropna(axis=0, how="all")

    df_return.isna().sum()
    df_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_return


# TEST 

# jeu de données issues du preprocessing successif
df_V2 =  pd.read_csv("../data_saved/data_preprocessed_V2.csv", index_col=["id_Location","id_Date"])
df_V2["Date"] = pd.to_datetime(df_V2["Date"])
df_V2["Location"] = df_V2.index.get_level_values(0)
df_V2 = df_V2.set_index(["Location", "Date"])
df_V2 = df_V2.reindex(df_V2.index.rename({"Location": "id_Location", "Date" : "id_Date"}))

df_V2.columns
location_list = df_V2.index.get_level_values(0).unique()
# création du dataframe avec toutes les dates 
df = df_V2

# choix des variables pour lesquelles on calcule 3 lags de 3 jours et une moyenne sur 3 jours
lag_vars = ["MinTemp", "MaxTemp", 
            "Evaporation", 'WindGustSpeed',
            'Sunshine', 'Humidity9am', 'Pressure9am', 'Temp9am',
            'Rainfall']

# variables pour lesquelles on calcule la variation entre 9am et 3pm
diff_vars = ["WindSpeed","Humidity","Pressure","Temp","Cloud"]

df_test = add_lagdelta_vars(df_V2, lag_vars, diff_vars)

df_V2.isna().sum()
df_test.isna().sum()
df_test.describe()