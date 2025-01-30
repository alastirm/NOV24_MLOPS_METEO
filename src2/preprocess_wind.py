import sys
import time
import threading
import warnings
import emoji

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin



class wind_speed_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_select: str = 'WindGustSpeed', col_target: str = 'RainTomorrow', method: str = 'median'):

        self.dict_wst = {}
        self.col_select = col_select
        self.col_target = col_target
        self.method = method


    def fit(self, X, y=None):

        self.dict_wst = X.groupby(self.col_target)[self.col_select].agg(self.method).to_dict()

        return self


    def transform(self, X):

        X[self.col_select] = X.apply(
            lambda row: self.dict_wst.get(row[self.col_target], row[self.col_select])
            if pd.isna(row[self.col_select]) else row[self.col_select],axis=1)

        X[self.col_select +'_mean_7'] = X[self.col_select].rolling(window=7, min_periods=1).mean()
        X[self.col_select + '_std_7'] = X[self.col_select].rolling(window=7, min_periods=1).std()
        X[self.col_select + '_max_7'] = X[self.col_select].rolling(window=7, min_periods=1).max()
        X[self.col_select + '_min_7'] = X[self.col_select].rolling(window=7, min_periods=1).min()

        X[self.col_select +'_mean_5'] = X[self.col_select].rolling(window=5, min_periods=1).mean()
        X[self.col_select + '_std_5'] = X[self.col_select].rolling(window=5, min_periods=1).std()
        X[self.col_select + '_max_5'] = X[self.col_select].rolling(window=5, min_periods=1).max()
        X[self.col_select + '_min_5'] = X[self.col_select].rolling(window=5, min_periods=1).min()

        X[self.col_select +'_mean_3'] = X[self.col_select].rolling(window=3, min_periods=1).mean()
        X[self.col_select + '_std_3'] = X[self.col_select].rolling(window=3, min_periods=1).std()
        X[self.col_select + '_max_3'] = X[self.col_select].rolling(window=3, min_periods=1).max()
        X[self.col_select + '_min_3'] = X[self.col_select].rolling(window=3, min_periods=1).min()

        X[self.col_select + '_diff'] = X[self.col_select].diff()

        return X




class wind_dir_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quanlitatives.
    elle prend en argument :
    geo : Location, mais peut être changer par geo,
    col_select : WindGustDir, colonne qualitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    Elle replace les Nan par la valeurs la plus fréquentes pour le couple geo et col_target
    '''

    def __init__(self, col_select : str = 'WindGustDir', col_target : str = 'RainTomorrow'):

        self.col_select = col_select
        self.col_target = col_target

        self.cardinal_mapping = {
            'N': 0,
            'NNE': 22.5,
            'NE': 45,
            'ENE': 67.5,
            'E': 90,
            'ESE': 112.5,
            'SE': 135,
            'SSE': 157.5,
            'S': 180,
            'SSW': 202.5,
            'SW': 225,
            'WSW': 247.5,
            'W': 270,
            'WNW': 292.5,
            'NW': 315,
            'NNW': 337.5,
        }

        self.cardinal_cos_sin = {}
        self.dict_wdt = {}

    def fit(self, X, y = None):
        # on met dans un dictionnaire, dont la clé est geo, les valeurs qui reviennent le plus souvent en fonction de col_select et col_target

        group = pd.DataFrame(X.groupby(self.col_target)[self.col_select].value_counts(ascending = False))

        self.dict_wdt[0] = group.loc[0].idxmax()[0]
        self.dict_wdt[1] = group.loc[0].idxmax()[0]


        for direction, angle in self.cardinal_mapping.items():
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))

            self.cardinal_cos_sin[direction] = (cos_a, sin_a)

        return self


    def transform(self, X):
        # choisit la valeur dans dict_wdt si col_select est Nan et en fonction de la valeur de col_target
        # sinon conserve la valeur de col_select

        X[self.col_select] = X.apply(
            lambda row: self.dict_wdt.get(row[self.col_target])
            if pd.isna(row[self.col_select]) else row[self.col_select],axis=1)

        X[self.col_select + '_cos'] = X[self.col_select].apply(lambda x : self.cardinal_cos_sin[x][0])
        X[self.col_select + '_sin'] = X[self.col_select].apply(lambda x : self.cardinal_cos_sin[x][1])



        return X
