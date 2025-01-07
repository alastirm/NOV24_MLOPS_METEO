import pandas as pd
import numpy as np

# Fonction qui initialise les données en fonction du chemin d'accès 'data_dir'
# Attention, nécessite le fichier climat_location_unique.csv


def initialize_data_weatherAU(data_dir):

    # Lecture dataset et infos
    df = pd.read_csv(filepath_or_buffer=data_dir, sep=",")

    # Ajout de la variable climate
    Location_climate = pd.read_csv("../data/Location_Climate_unique.csv")
    df = pd.merge(df, Location_climate,  on="Location", how="left")

    # Transforme la date en datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # discrétisation variable cible -> à mettre dans preprocess
    df['RainTomorrow'] = df['RainTomorrow'].replace(['Yes', 'No'], [1, 0])
    # discrétisation variable RainToday -> à mettre dans preprocess
    df['RainToday'] = df['RainToday'].replace(['Yes', 'No'], [1, 0])

    # # Met la date en indice avec la station mais sans retirer les colonnes pour le moment
    df = df.set_index(["Location", "Date"], drop=False)
    df = df.reindex(df.index.rename({"Location" : "id_Location", "Date" : "id_Date"}))

    return df
