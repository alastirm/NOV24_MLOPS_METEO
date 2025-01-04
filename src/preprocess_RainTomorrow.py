import pandas as pd
import numpy as np


from functions_created import create_date_list

# Fonction pour remplir les Nas de RainTomorrow avec les valeurs de RainToday pour le jour suivant

def fillNa_RainTommorrow(df):
    # Compte le nombre de Nas remplacés
    count = 0
    # boucle sur les locations

    for location in df["Location"].unique():
        print(location)

        # création listes de dates
        df_dates, date_list = create_date_list(df)

        # création d'un df avec toutes les dates
        df_location = df_dates[["Dates", "Year"]]
        df_location["Location"] = location
        df_location = df_location.set_index(["Location", "Dates"])
        df_location = df_location.reindex(df_location.index.rename({"Location": "id_Location", "Dates" : "id_Date"}))

        # ajout de la colonne RainTomorrow pour cette location
        df_location = pd.merge(df_location, df["RainTomorrow"],
                               left_index=True , right_index=True, how="left")

        # Ajout de la colonne RainToday décalée d'un jour 
        # (i.e devrait être la même que RainTomorrow)
        df_location = pd.merge(df_location, 
                               df.loc[(location)]["RainToday"].shift(-1, freq="D"),
                               left_index=True, right_index=True, how="left")

        # df_location[(df_location["RainTomorrow"].isna() == False)]

        # Si RainToday n'est pas manquant pour le jour d'après, 
        # on impute RainTomorrox de la veille (t)

        data_to_replace = df_location[(df_location["RainTomorrow"].isna()) &
                                      (~ (df_location["RainToday"].isna()))]

        count += len(data_to_replace)
        print("Dates remplacées : ", data_to_replace.index)

        df_location.loc[(df_location["RainTomorrow"].isna()) &
               (~ (df_location["RainToday"].isna())), ["RainTomorrow"]] = \
            df_location.loc[(df_location["RainTomorrow"].isna()) &
                   (~ (df_location["RainToday"].isna())),["RainToday"]]
        
        df_location = df_location.rename(columns={"RainTomorrow": "RainTomorrow_new"})
        
        df_location = df_location["RainTomorrow_new"]

        if location == df["Location"].unique()[0]:
            df_new = df_location
        else:
            df_new = pd.concat([df_new, df_location])
    
    print("Remplacement de ", count, "Nas sur la variable RainTomorrow")
    return df_new


def preprocess_RainTomorrow(df):
    # discrétisation variable cible
    df['RainTomorrow'] = df['RainTomorrow'].replace(['Yes', 'No'], [1,0])

    # Remplissage des Nas avec les valeurs de RainToday pour le jour suivant
    df_new = fillNa_RainTommorrow(df)
    df = pd.merge(df, df_new, left_index=True, right_index=True)
    df["RainTomorrow"] = df["RainTomorrow_new"]
    df = df.drop(columns="RainTomorrow_new")

    # Suppression des Nas restants pour la variable cible RainTomorrow
    df = df.dropna(subset = "RainTomorrow")
    
    return df

