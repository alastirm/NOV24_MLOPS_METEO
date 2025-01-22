import pandas as pd
import numpy as np
import emoji

def preprocess_RainToday(df):
    # discrétisation variable RainToday
    df['RainToday'] = df['RainToday'].replace(['Yes', 'No'], [1, 0])

    # Modification de la variable RainToday après remplissage des Nas de la variable RainFall

    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=1),"RainToday"] = 1
    df.loc[(df["RainToday"].isna()) & (df["Rainfall"] >=0) &
       (df["Rainfall"] <1), "RainToday"] = 0

    print(df["RainToday"].isna().sum())
    print(f"encodage {"RainToday"} {emoji.emojize(':thumbs_up:')}")

    return df


