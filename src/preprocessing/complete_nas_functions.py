import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import sys
import time
import threading
import warnings
import emoji

from sklearn.preprocessing import FunctionTransformer

import geopy.distance 

def create_distance_matrix():
    # coordonnées des stations

    cities_coordinates = {
        'Albury': (-36.0734, 146.9169),
        'BadgerysCreek': (-33.8891, 150.7725),
        'Cobar': (-31.8647, 145.8225),
        'CoffsHarbour': (-30.2989, 153.1145),
        'Moree': (-29.4736, 149.8411),
        'Newcastle': (-32.9283, 151.7817),
        'NorahHead': (-33.1920, 151.5216),
        'NorfolkIsland': (-29.0408, 167.9591),
        'Penrith': (-33.7491, 150.6941),
        'Richmond': (-33.5990, 150.7670),
        'Sydney': (-33.8688, 151.2093),
        'SydneyAirport': (-33.9399, 151.1753),
        'WaggaWagga': (-35.1100, 147.3673),
        'Williamtown': (-32.7990, 151.8430),
        'Wollongong': (-34.4278, 150.8931),
        'Canberra': (-35.2809, 149.1300),
        'Tuggeranong': (-35.4135, 149.0705),
        'MountGinini': (-35.4800, 148.9325),
        'Ballarat': (-37.5622, 143.8503),
        'Bendigo': (-36.7580, 144.2805),
        'Sale': (-38.1000, 147.0667),
        'MelbourneAirport': (-37.6733, 144.8431),
        'Melbourne': (-37.8136, 144.9631),
        'Mildura': (-34.1842, 142.1593),
        'Nhil': (-35.6544, 141.6931),
        'Portland': (-38.3553, 141.5883),
        'Watsonia': (-37.7170, 145.1265),
        'Dartmoor': (-37.8689, 141.4203),
        'Brisbane': (-27.4698, 153.0251),
        'Cairns': (-16.9203, 145.7710),
        'GoldCoast': (-28.0167, 153.4000),
        'Townsville': (-19.2589, 146.8184),
        'Adelaide': (-34.9285, 138.6007),
        'MountGambier': (-37.8333, 140.7833),
        'Nuriootpa': (-34.4934, 138.9773),
        'Woomera': (-31.1333, 136.8333),
        'Albany': (-35.0200, 117.8833),
        'Witchcliffe': (-33.7969, 115.1236),
        'PearceRAAF': (-31.8036, 115.9747),
        'PerthAirport': (-31.9385, 115.9773),
        'Perth': (-31.9505, 115.8605),
        'SalmonGums': (-33.0091, 120.3702),
        'Walpole': (-34.9789, 116.6293),
        'Hobart': (-42.8821, 147.3272),
        'Launceston': (-41.4403, 147.1349),
        'AliceSprings': (-23.6980, 133.8807),
        'Darwin': (-12.4634, 130.8456),
        'Katherine': (-14.4683, 132.2615),
        'Uluru': (-25.3444, 131.0369)
    }

    location_list = list(cities_coordinates.keys())
    dist_mat = pd.DataFrame(index = location_list)
    
    for location in location_list:
        for location2 in location_list:
            dist_mat.loc[location, location2] = \
                geopy.distance.distance(cities_coordinates[location], 
                                        cities_coordinates[location2]).km
            
        
    return dist_mat

def complete_na_neareststation(df, variable, distance_max):
    dist_mat = create_distance_matrix()
    print("récupération de valeurs à moins de ", distance_max, " km pour la variable", variable)
    
    # On récupère les dates manquantes pour la variable
    df_na_variable = df[(df[variable].isna())][["Date", "Location", variable]]
    location_list = df_na_variable.Location.unique()
    df_na_variable[variable + "_near"] = np.nan

    total_count = 0

    # recherche par station de valeurs dans la station la plus proche

    for location in location_list:
        # print("Station :", location)
        # Stations à moins de  distance_max km
        nearest_stations = dist_mat.loc[location, dist_mat.loc[location] <   distance_max]
        nearest_stations = nearest_stations[nearest_stations.index != location]
        count = 0

        if len(nearest_stations) == 0:
        #     print("Pas de station à moins de ", distance_max, " km de ", location)
            count = count
        else:
            # On garde la plus proche 
            station_check = nearest_stations.sort_values().index[0]
            dates_to_check = list(df_na_variable.loc[(df_na_variable["Location"] == location),"Date"])
            # print("Station la plus proche : ", station_check)
            
            # boucles sur les dates Nas pour récupérer les valeurs de la station à proximité
            for date_tmp in dates_to_check:
                # print(date_tmp)
                # On ne teste que les dates qui existent aussi pour la station la plus proche
                if date_tmp in df.loc[df["Location"] == station_check].index.get_level_values(1):
                    # Si une valeur existe, on l'impute
                    if df.loc[(station_check, date_tmp), variable] != np.nan:
                        df_na_variable.loc[(location, date_tmp), variable + "_near"] = \
                            df.loc[(station_check, date_tmp), variable]
                        
                        count += 1
            
        # print(count, " valeurs récupérées pour la variable ", variable, "et la station", location)
        total_count += count
        
    print(total_count, "Valeurs récupérées pour la variable ", variable, f"{emoji.emojize(':thumbs_up:')}")
    
    # Ajout des valeurs récupérées au dataframe de base
    df = pd.merge(df, df_na_variable[variable + "_near"], left_index= True, right_index= True, how = "left")

    df.loc[(df[variable].isna()) & ~ (df[variable + "_near"].isna()),variable] = \
        df.loc[(df[variable].isna()) & ~ (df[variable + "_near"].isna()),variable + "_near"]

    # Nas restants
    print(df[variable].isna().sum(), "Nas restants pour la variable", variable)
    # suppression de la nouvelle colonne 
    df = df.drop(columns = variable + "_near")

    return df


# Fonction pour appliquer la médiane selon la localisation et la saison
def complete_na_median_location(df, columns):
    median_values = df.groupby(["Season", "Location"])[columns].median()
    for col in columns:
        df[col] = df.set_index(["Season", "Location"])[col].fillna(median_values[col]).values
    return df

# Fonction pour appliquer la médiane selon le climat et la saison
def complete_na_median_climate(df, columns):
    median_values = df.groupby(["Season", "Climate"])[columns].median()
    for col in columns:
        df[col] = df.set_index(["Season", "Climate"])[col].fillna(median_values[col]).values
    return df

# Fonction pour appliquer la moyenne des valeurs de la veille et du lendemain
def complete_na_mean(df, columns):
    for col in columns:
        df[col] = df[col].interpolate(method="linear")
    return df

def complete_na_median_location_month(df, columns):
    
    median_values = df.groupby(["Month", "Location"])[columns].median()
    for col in columns:
        df[col] = df.set_index(["Month", "Location"])[col].fillna(median_values[col]).values
    
    print("Remplacement de valeurs manquantes par la médiane par station et mois pour la variable ", columns, f"{emoji.emojize(':thumbs_up:')}")
    return df

def complete_na_mode_location_month(df, columns):
    
    mode_values = df.groupby(["Month", "Location"])[columns].agg(lambda x: pd.Series.mode(x, dropna=False)[0])
    for col in columns:
        df[col] = df.set_index(["Month", "Location"])[col].fillna(mode_values[col]).values
    
    print("Remplacement de valeurs manquantes par le mode par station et mois pour la variable ", columns, f"{emoji.emojize(':thumbs_up:')}")
    return df

def complete_na_mode_location_target(df, columns):
    
    mode_values = df.groupby(["Location", "RainTomorrow"])[columns].agg(lambda x: pd.Series.mode(x, dropna=False)[0])
    mode_values.replace('[]',np.nan)
    for col in columns:
        df[col] = df.set_index(["Location", "RainTomorrow"])[col].fillna(mode_values[col]).values
    


    print("Remplacement de valeurs manquantes par le mode par station et mois pour la variable ", columns, f"{emoji.emojize(':thumbs_up:')}")
    return df


class quanti_geo_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quantitatives.
    elle prend en argument :
    geo : Climate, mais peut être changer par location,
    col_select : windgustspeed, colonne quantitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    method : median, peut être remplacer par mean, mod, etc
    '''


    def __init__(self, col_select, geo :str = 'Climate', col_target : str = 'RainTomorrow', method : str = 'median'):

        self.dict_wst = {}
        self.geo = geo
        self.col_select = col_select
        self.col_target = col_target
        self.method = method

    def fit(self, X, y = None):
        # on récupère dans un dictionnaire la valeur de method pour la col_select dont on veut gérer les Nan, avec clé tuple(geo, col_target)

        for i in X[self.geo].unique():
            for j in X[self.col_target].unique():
                self.dict_wst[i, j] = getattr(X[(X[self.geo] == i) & (X[self.col_target] == j)][self.col_select], self.method)()

        return self

    def transform(self, X):
        # on cherche le tuple(geo, col_target) et on applique la valeur de la method si c'est un Nan

        X[self.col_select] = X.apply(
                lambda row: self.dict_wst.get((row[self.geo], row[self.col_target]), row[self.col_select])
                if pd.isna(row[self.col_select]) else row[self.col_select],
                axis=1)

        print(f"Remplacement Na {self.col_select} {emoji.emojize(':thumbs_up:')}")

        return X



class quali_geo_target_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quanlitatives.
    elle prend en argument :
    geo : Location, mais peut être changer par geo,
    col_select : WindGustDir, colonne qualitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    Elle replace les Nan par la valeurs la plus fréquentes pour le couple geo et col_target
    '''

    def __init__(self, col_select, geo : str = 'Location',  col_target : str = 'RainTomorrow'):

        self.dict_wdt = {}
        self.geo = geo
        self.col_select = col_select
        self.col_target = col_target

    def fit(self, X, y = None):
        # on met dans un dictionnaire, dont la clé est geo, les valeurs qui reviennent le plus souvent en fonction de col_select et col_target

        for i in X[self.geo].unique():
            self.dict_wdt[i] = []
            mat = X[X[self.geo] == i].groupby(self.col_target)[self.col_select].value_counts(ascending=False).unstack()
            for j in X[self.col_target].unique():
                if j in mat.index:
                    self.dict_wdt[i].append(mat.loc[j].idxmax())

        return self

    def transform(self, X):
        # choisit la valeur dans dict_wdt si col_select est Nan et en fonction de la valeur de col_target
        # sinon conserve la valeur de col_select

        X[self.col_select] = X.apply(
            lambda row: self.dict_wdt[row[self.geo]][0] if pd.isna(row[self.col_select]) and row[self.col_target] == 0
                  else self.dict_wdt[row[self.geo]][1] if pd.isna(row[self.col_select]) and row[self.col_target] == 1
                  else row[self.col_select],
            axis=1)

        print(f"Remplacement Na {self.col_select} {emoji.emojize(':thumbs_up:')}")

        return X

def create_complete_nas_pipeline(cols_to_fill, preprocess_method, distance_max=50):
    
    complete_nas_transformer = Pipeline(steps=[('init','')])
    
    if preprocess_method == "nearest": 
        for col_select in cols_to_fill:
            complete_nas_transformer.steps.append((('nearest_nas_' + col_select), 
             FunctionTransformer(complete_na_neareststation, 
                                 kw_args={'distance_max':distance_max, 
                                          'variable':col_select})))
            
    if preprocess_method == "median_location_month":
        for col_select in cols_to_fill:
            complete_nas_transformer.steps.append((('median_location_month_' + col_select), 
             FunctionTransformer(complete_na_median_location_month,
                                 kw_args={'columns':[col_select]})))
    
    if preprocess_method == "mode_location_month":
        for col_select in cols_to_fill:
             complete_nas_transformer.steps.append((('mode_location_month_' + col_select), 
             FunctionTransformer(complete_na_mode_location_month,
                                 kw_args={'columns':[col_select]})))

    if preprocess_method == "mode_location_target":
        for col_select in cols_to_fill:
             complete_nas_transformer.steps.append((('mode_location_target_' + col_select), 
             FunctionTransformer(complete_na_mode_location_target,
                                 kw_args={'columns':[col_select]})))
             
    if preprocess_method == "nearest_median_location_month":

        for col_select in cols_to_fill:
            complete_nas_transformer.steps.append(
            (('nearest_nas_' + col_select), 
             FunctionTransformer(complete_na_neareststation, 
                                 kw_args={'distance_max':distance_max, 
                                          'variable':col_select})))
            complete_nas_transformer.steps.append(
            (('median_location_month_' + col_select), 
             FunctionTransformer(complete_na_median_location_month,
                                 kw_args={'columns':[col_select]})))
            
    complete_nas_transformer.steps.pop(0)
    
    return complete_nas_transformer
