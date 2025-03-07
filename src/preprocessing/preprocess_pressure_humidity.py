import pandas as pd

def remplissage_na_voisinnage(location, column, global_data):
    
    i = 0
    location_len = len(location)
    while i < location_len:
        if pd.isna(location[column].iloc[i]):
            periode = 0
            while (i+periode < location_len) and (pd.isna(location[column].iloc[i+periode])):
                periode += 1
            
            ratio_na = location.isna().mean()
            if periode == location_len:
                location[column] = global_data.median()
                i += location_len
            else:
                borne_sup = min(location_len-1, i+periode)
                date_debut = location["Date"].iloc[i]
                date_fin = location["Date"].iloc[borne_sup]
                nombre_jours = (date_fin - date_debut).days + 1

                voisinnage = location[(location["Date"] >= date_debut - pd.Timedelta(days=nombre_jours)) &
                                    (location["Date"] <= date_fin + pd.Timedelta(days=nombre_jours)) &
                                    (~location["Date"].isna())]

                if not voisinnage.empty:
                    location.loc[location.index[i]:location.index[borne_sup], column] = voisinnage[column].mean()
                else:
                    location[column] = global_data.median()
                i += periode
        
        else:
            i += 1
    return location

def remplir_na(df, columns=["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"]):

    for col in columns:
        df = df.groupby("Location").apply(remplissage_na_voisinnage, column=col, global_data=df[col])