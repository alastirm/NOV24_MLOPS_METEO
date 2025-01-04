import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Fonction qui extrait la saison en Australie pour une variable au format Datetime
# x doit contenir le mois

def get_season_AU(x):

    if (x.month, x.day) < (3, 20) or (x.month, x.day) > (12, 20):
        return 'Summer'
    elif (x.month, x.day) < (6, 21):
        return 'Autumn'
    elif (x.month, x.day) < (9, 20):
        return 'Winter'
    elif (x.month, x.day) <= (12, 20):
        return 'Spring'
    else:
        raise IndexError("Invalid Input")

def preprocess_Date(df):

    # Transformation en datetime et création des variables mois, années
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

  
    # Modification variable month pour avoir le bon ordre dans les graphes
    mois_en = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df["Month"] = pd.Categorical(
        df["Month"],
        categories=mois_en,
        ordered=True)

    # Ajout de la variable Season
    df["Season"] = df["Date"].apply(lambda x: get_season_AU(x))
    return df


def encode_Date(df):

    # Encodage des variables 
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,  dtype='int')
    
    # Saison : dummies par saison
    # Mois : dummies par saison
    # Year : dummies par saison

    # On fit l'encodeur sur les variables voulues
    vars_to_encode = ["Season","Year","Month"]
    encoder.fit(df[vars_to_encode])

    # On crée les dummies
    var_enc = encoder.transform(df[vars_to_encode])
    var_enc = pd.DataFrame(var_enc, 
                           columns= encoder.get_feature_names_out(), 
                           index = df.index)

    ## On ajoute les variables encodées au df de base et on supprime les variables d'origine
    df = pd.merge(df, var_enc,
                  left_index = True, right_index=True)
    
    df =  df.drop(columns = vars_to_encode)
    return df


