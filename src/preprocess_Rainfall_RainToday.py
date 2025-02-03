import pandas as pd
import numpy as np

import functions_created


def preprocess_Rainfall_RainToday(df):
    # discrétisation variable RainToday
    df['RainToday'] = df['RainToday'].replace(['Yes', 'No'], [1, 0])

    # remplissage des Nas par les valeurs de la station à proximité le même jour (moins de 50 km)

    Rainfall_near = functions_created.complete_na_neareststation(df, 
                                                                 variable="Rainfall",  
                                                                 distance_max=50)

    # Ajout des valeurs récupérées au dataframe

    df = pd.merge(df, Rainfall_near["Rainfall_near"], left_index= True, right_index= True, how = "left")

    df.loc[(df["Rainfall"].isna()) & ~ (df["Rainfall_near"].isna()),"Rainfall"] = \
        df.loc[(df["Rainfall"].isna()) & ~ (df["Rainfall_near"].isna()),"Rainfall_near"]

    df["Rainfall"].isna().sum()
    df[~(df["RainTomorrow"].isna())]["Rainfall"].isna().sum()

    # suppression de la nouvelle colonne 
    df = df.drop(columns = "Rainfall_near")

    # Modification de la variable RainToday en conséquence

    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=1),"RainToday"] = 1
    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=0) &
       (df["Rainfall"] <1), "RainToday"] = 0

    df["RainToday"].isna().sum()

    # On supprime les Nas restants

    df = df.dropna(subset = ["Rainfall","RainToday"])

    return df


