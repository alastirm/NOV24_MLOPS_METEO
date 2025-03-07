import requests
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from pathlib import Path

def preprocess_scrap_data() :

    station_ID = pd.read_csv("../data/station_ID.csv", sep=",")

    # drop les stations sans ID
    station_ID = station_ID.dropna(subset = ["IDCJDW"])
    station_ID = station_ID.dropna(subset = ["IDCJDW"])

    # drop les nouvelles stations pour le moment
    station_ID = station_ID.dropna(subset = ["Location"])
    station_ID["IDCJDW"] = station_ID["IDCJDW"].astype(int).astype(str)
    station_ID.head()

    ## time list 

    year_list = ["2024","2025"]
    month_list = {"2024": ["01","02","03","04","05","06","07","08","09","10","11","12"],
                "2025": ["01","02"]}

    df_global = pd.DataFrame()
    for location in station_ID.Location2:
        df_location = pd.DataFrame()
        print("Preprocessing new data on ", location)
        for year in year_list :
            for month in month_list[year] :
                dir_csv = "../data/scrapcsv/" + location + "_" + year + month + ".csv"
                if Path(dir_csv).exists():
                    df = pd.read_csv(dir_csv, header = 0).drop(columns = 'Unnamed: 0')
                    df_location = pd.concat([df_location, df])        
            print("year ", year, " OK")
        df_location["Location"] = location
        df_location.rename(columns = {
        'Minimum temperature (°C)' : 'MinTemp',
        'Maximum temperature (°C)' : 'MaxTemp',
        'Rainfall (mm)' : 'Rainfall',
        'Evaporation (mm)' : 'Evaporation',
        'Sunshine (hours)' : 'Sunshine',
        'Direction of maximum wind gust ' : 'WindGustDir',
        'Speed of maximum wind gust (km/h)' : 'WindGustSpeed',
        '9am Temperature (°C)' : 'Temp9am',
        '9am relative humidity (%)' : 'Humidity9am',
        '9am cloud amount (oktas)' : 'Cloud9am',
        '9am wind direction' : 'WindDir9am',
        '9am wind speed (km/h)' : 'WindSpeed9am',
        '9am MSL pressure (hPa)' : 'Pressure9am',
        '3pm Temperature (°C)' : 'Temp3pm',
        '3pm relative humidity (%)' : 'Humidity3pm',
        '3pm cloud amount (oktas)' : 'Cloud3pm',
        '3pm wind direction' : 'WindDir3pm',
        '3pm wind speed (km/h)' : 'WindSpeed3pm',
        '3pm MSL pressure (hPa)' : 'Pressure3pm',
        }, inplace = True)
        
        
        df_location.drop(columns = ['Time of maximum wind gust'], 
            inplace = True)
        
        df_location['RainToday'] = df_location['Rainfall'].apply(lambda x: 'Yes' if x >= 1 else 'No')
        df_location['RainTomorrow'] = df_location['RainToday'].shift(-1)
        df_location['Humidity3pm'] = df_location['Humidity3pm'].astype(float)
        df_location['WindSpeed3pm'] = df_location['WindSpeed3pm'].apply(lambda x : 0 if x == 'Calm' else x)
        df_location['WindSpeed9am'] = df_location['WindSpeed9am'].apply(lambda x : 0 if x == 'Calm' else x)
        df_location['Humidity3pm'] = df_location['Humidity3pm'].astype(float)
        df_location['Humidity9am'] = df_location['Humidity9am'].astype(float)
        df_location['WindSpeed3pm'] = df_location['WindSpeed3pm'].astype(float)
        df_location['WindSpeed9am'] = df_location['WindSpeed9am'].astype(float)
        df_location['WindDir9am'] = df_location['WindDir9am'].replace(' ', pd.NA)
        df_location['WindDir9am'] = df_location['WindDir9am'].ffill()
        df_location['WindDir3pm'] = df_location['WindDir3pm'].replace(' ', pd.NA)
        df_location['WindDir3pm'] = df_location['WindDir3pm'].ffill()

        df_global = pd.concat([df_global, df_location])


    # Ajout de la variable climate
    Location_climate = pd.read_csv("../data/Location_Climate_unique.csv")
    df_global = pd.merge(df_global, Location_climate,  on="Location", how="left")

    # Transforme la date en datetime
    df_global["Date"] = pd.to_datetime(df_global["Date"])

    # discrétisation variable cible -> à mettre dans preprocess
    df_global['RainTomorrow'] = df_global['RainTomorrow'].replace(['Yes', 'No'], [1, 0])
    # discrétisation variable RainToday -> à mettre dans preprocess
    df_global['RainToday'] = df_global['RainToday'].replace(['Yes', 'No'], [1, 0])

    # # Met la date en indice avec la station mais sans retirer les colonnes pour le moment
    df_global = df_global.set_index(["Location", "Date"], drop=False)
    df_global = df_global.reindex(df_global.index.rename({"Location" : "id_Location", "Date" : "id_Date"}))

    return df_global




def add_scrap_data(new_data_name) :

    station_ID = pd.read_csv("../data/station_ID.csv", sep=",")

    # drop les stations sans ID
    station_ID = station_ID.dropna(subset = ["IDCJDW"])
    station_ID = station_ID.dropna(subset = ["IDCJDW"])

    # drop les nouvelles stations pour le moment
    station_ID = station_ID.dropna(subset = ["Location"])
    station_ID["IDCJDW"] = station_ID["IDCJDW"].astype(int).astype(str)
    station_ID.head()

    ## time list 

    year_list = ["2024","2025"]
    month_list = {"2024": ["01","02","03","04","05","06","07","08","09","10","11","12"],
                "2025": ["01","02"]}

    df_global = pd.DataFrame()
    for location in station_ID.Location2:
        df_location = pd.DataFrame()
        print("Preprocessing new data on ", location)
        for year in year_list :
            for month in month_list[year] :
                dir_csv = "../data/scrapcsv/" + location + "_" + year + month + ".csv"
                if Path(dir_csv).exists():
                    df = pd.read_csv(dir_csv, header = 0).drop(columns = 'Unnamed: 0')
                    df_location = pd.concat([df_location, df])        
            print("year ", year, " OK")
        df_location["Location"] = location
        df_location.rename(columns = {
        'Minimum temperature (°C)' : 'MinTemp',
        'Maximum temperature (°C)' : 'MaxTemp',
        'Rainfall (mm)' : 'Rainfall',
        'Evaporation (mm)' : 'Evaporation',
        'Sunshine (hours)' : 'Sunshine',
        'Direction of maximum wind gust ' : 'WindGustDir',
        'Speed of maximum wind gust (km/h)' : 'WindGustSpeed',
        '9am Temperature (°C)' : 'Temp9am',
        '9am relative humidity (%)' : 'Humidity9am',
        '9am cloud amount (oktas)' : 'Cloud9am',
        '9am wind direction' : 'WindDir9am',
        '9am wind speed (km/h)' : 'WindSpeed9am',
        '9am MSL pressure (hPa)' : 'Pressure9am',
        '3pm Temperature (°C)' : 'Temp3pm',
        '3pm relative humidity (%)' : 'Humidity3pm',
        '3pm cloud amount (oktas)' : 'Cloud3pm',
        '3pm wind direction' : 'WindDir3pm',
        '3pm wind speed (km/h)' : 'WindSpeed3pm',
        '3pm MSL pressure (hPa)' : 'Pressure3pm',
        }, inplace = True)
        
        
        df_location.drop(columns = ['Time of maximum wind gust'], 
            inplace = True)
        
        df_location['RainToday'] = df_location['Rainfall'].apply(lambda x: 'Yes' if x >= 1 else 'No')
        df_location['RainTomorrow'] = df_location['RainToday'].shift(-1)
        df_location['Humidity3pm'] = df_location['Humidity3pm'].astype(float)
        df_location['WindSpeed3pm'] = df_location['WindSpeed3pm'].apply(lambda x : 0 if x == 'Calm' else x)
        df_location['WindSpeed9am'] = df_location['WindSpeed9am'].apply(lambda x : 0 if x == 'Calm' else x)
        df_location['Humidity3pm'] = df_location['Humidity3pm'].astype(float)
        df_location['Humidity9am'] = df_location['Humidity9am'].astype(float)
        df_location['WindSpeed3pm'] = df_location['WindSpeed3pm'].astype(float)
        df_location['WindSpeed9am'] = df_location['WindSpeed9am'].astype(float)
        df_location['WindDir9am'] = df_location['WindDir9am'].replace(' ', pd.NA)
        df_location['WindDir9am'] = df_location['WindDir9am'].ffill()
        df_location['WindDir3pm'] = df_location['WindDir3pm'].replace(' ', pd.NA)
        df_location['WindDir3pm'] = df_location['WindDir3pm'].ffill()

        df_global = pd.concat([df_global, df_location])


    # chargement des données initiales
    data_dir = "../data/weatherAUS.csv"
    df_init = pd.read_csv(data_dir)
    
    # concaténation deux données
    df_tuned = pd.concat([df_init, df_global])
    df_tuned.to_csv("../data/"+ new_data_name +".csv")
    print("Dataframe créé !")
    return df_tuned