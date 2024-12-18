import pandas as pd
import numpy as np


# Fonction qui extrait la saison en Australie pour une variable au format Datetime
# x doit contenir le mois et le jour

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
    ordered=True
    )

    # Ajout de la variable Season
    df["Season"] = df["Date"].apply(lambda x : get_season_AU(x))
    return df
