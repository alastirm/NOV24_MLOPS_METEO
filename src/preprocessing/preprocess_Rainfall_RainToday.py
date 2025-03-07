import pandas as pd
import numpy as np

from complete_nas_functions import complete_na_neareststation

def preprocess_Rainfall_RainToday(df):
    # discrétisation variable RainToday
    df['RainToday'] = df['RainToday'].replace(['Yes', 'No'], [1, 0])

    # remplissage des Nas par les valeurs de la station à proximité le même jour (moins de 50 km)

    df = complete_na_neareststation(df, variable="Rainfall", distance_max=50) 
                                                                  
    # Modification de la variable RainToday en conséquence

    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=1),"RainToday"] = 1
    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=0) &
       (df["Rainfall"] <1), "RainToday"] = 0

    df["RainToday"].isna().sum()

    # On supprime les Nas restants

    df = df.dropna(subset = ["Rainfall","RainToday"])

    return df


