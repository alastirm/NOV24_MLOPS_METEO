import sys
import time
import threading
import warnings
import emoji

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class wind_speed_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quantitatives.
    elle prend en argument :
    geo : Climate, mais peut être changer par location,
    col_select : windgustspeed, colonne quantitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    method : median, peut être remplacer par mean, mod, etc
    '''


    def __init__(self, geo :str = 'Climate', col_select : str = 'WindGustSpeed', col_target : str = 'RainTomorrow', method : str = 'median'):

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



class wind_dir_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quanlitatives.
    elle prend en argument :
    geo : Location, mais peut être changer par geo,
    col_select : WindGustDir, colonne qualitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    Elle replace les Nan par la valeurs la plus fréquentes pour le couple geo et col_target
    '''

    def __init__(self, geo : str = 'Location', col_select : str = 'WindGustDir', col_target : str = 'RainTomorrow'):

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


class compass_dir_encoder(BaseEstimator, TransformerMixin):

    def __init__(self, col_select : str = 'WindGustDir'):

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

        self.col_select = col_select
        self.cardinal_cos_sin = {}


    def fit(self, X, y=None):

        for direction, angle in self.cardinal_mapping.items():
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))

            self.cardinal_cos_sin[direction] = (cos_a, sin_a)

        return self


    def transform(self, X):

        X[self.col_select + '_cos'] = X[self.col_select].apply(lambda x : self.cardinal_cos_sin[x][0])
        X[self.col_select + '_sin'] = X[self.col_select].apply(lambda x : self.cardinal_cos_sin[x][1])

        print(f"encodage {self.col_select} {emoji.emojize(':thumbs_up:')}")

        return X




######################################################
######################################################





def pipeline_wind_function(df):

    warnings.filterwarnings('ignore')

    print('\n\ndébut du preprocessing des colonnes Wind...', end = '\n\n')
    # print('----> avant : ', df.isna().sum(), end = '\n\n')


    df = df[(df['Location'] != 'Newcastle') & (df['Location'] != 'Albany')]


    pipeline_wind = Pipeline([
        ('windgustspeed_transformer', wind_speed_transformer(col_select='WindGustSpeed')),
        ('windspeed9am_transformer', wind_speed_transformer(col_select='WindSpeed9am')),
        ('windspeed3pm_transformer', wind_speed_transformer(col_select='WindSpeed3pm')),
        ('windgustdir_tranformer', wind_dir_transformer(col_select = 'WindGustDir')),
        ('windspeed9am_tranformer', wind_dir_transformer(col_select = 'WindDir9am')),
        ('windspeed3pm_tranformer', wind_dir_transformer(col_select = 'WindDir3pm')),
        ('windgustdir_trigo_encoder', compass_dir_encoder(col_select = "WindGustDir")),
        ('winddir9am_trigo_encod', compass_dir_encoder(col_select = "WindDir9am")),
        ('winddir3pm_trigo_encod', compass_dir_encoder(col_select = "WindDir3pm"))
    ])

    pipeline_wind.fit_transform(df)

    df.drop(columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm'], inplace = True)

    print(emoji.emojize('\n ...fin du preprocessing des colonnes Wind :saluting_face: :saluting_face: :saluting_face: :saluting_face: :saluting_face: :saluting_face:'), end = '\n\n')
    # print('----> après', df.isna().sum())

    return df


# if __name__ == "__main__":

#     df = pd.read_csv('../data/weatherAUS.csv')
#     climate = pd.read_csv('../data/Location_Climate_unique.csv').set_index('Location')['Climate'].to_dict()
#     df = pd.read_csv('../data/weatherAUS.csv')
#     df['Climate'] = df['Location'].map(climate)
#     df = df.dropna(subset = 'RainTomorrow')
#     df = df[(df['Location'] != 'Newcastle') & (df['Location'] != 'Albany')]
#     df['RainTomorrow'] = df['RainTomorrow'].replace(['Yes', 'No'], [1, 0])

#     pipeline_wind_function(df)
