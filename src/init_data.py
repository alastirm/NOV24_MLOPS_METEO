import pandas as pd
import numpy as np

# Fonction qui initialise les données en fonction du chemin d'accès 'data_dir'
# Attention, nécessite le fichier climat_location_unique.csv 

def initialize_data_weatherAU(data_dir):
    
    # Lecture dataset et infos 
    df = pd.read_csv(filepath_or_buffer=data_dir, sep=",")
    
    # Suppression des Nas pour la variable cible RainTomorrow
    df = df.dropna(subset = "RainTomorrow")
    
    # discrétisation variable cible
    df['RainTomorrow'] = df['RainTomorrow'].replace(['Yes', 'No'], [1,0])
    
    # Ajout de la variable climate 
    Location_climate = pd.read_csv("../data_csv/climat_location_unique.csv")
    df = pd.merge(df, Location_climate,  on="Location", how="left")
    
    return df

