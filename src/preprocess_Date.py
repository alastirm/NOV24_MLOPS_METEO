import pandas as pd
import numpy as np

def preprocess_Date(df):

    # Transformation en datetime et création des variables mois, années
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()

    # Modification variable month pour avoir le bon ordre
    mois_en = ['January','February','March','April','May','June','July','August','September','October','November','December']
    df["Month"] = pd.Categorical(
    df["Month"] ,
    categories=mois_en,
    ordered=True
    )

    return df
