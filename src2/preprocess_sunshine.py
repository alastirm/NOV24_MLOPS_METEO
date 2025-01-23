import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class sunshine_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quantitatives.
    elle prend en argument :
    geo : location, mais peut être changer par location,
    col_select : sunshine, colonne quantitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    method : median, peut être remplacer par mean, mod, etc
    '''


    def __init__(self, geo :str = 'Location', col_select : str = 'Sunshine', col_target : str = 'RainTomorrow', method : str = 'median'):
        self.dict_sun = {}
        self.geo = geo
        self.col_select = col_select
        self.col_target = col_target
        self.method = method

    def fit(self, X, y = None):
        # on récupère dans un dictionnaire la valeur de method pour la col_select dont on veut gérer les Nan, avec clé tuple(geo, col_target)

        X_drop = X.copy()
        X_drop = X_drop.dropna(subset = self.col_select)
        X_drop = pd.concat([X['Location'].value_counts(), X_drop['Location'].value_counts()], axis = 1)
        X_drop.columns = ['Location_df', 'Location_df_sun']
        X_drop['ratio'] = (X_drop['Location_df_sun'] / X_drop['Location_df'])
        X_drop = X_drop[X_drop['ratio'] > 0.7]

        for i in X_drop.index:
            for j in X[self.col_target].unique():
                self.dict_sun[i, j] = X[(X[self.geo] == i) & (X[self.col_target] == j)][self.col_select].agg(self.method)

        return self

    def transform(self, X):
        # on cherche le tuple(geo, col_target) et on applique la valeur de la method si c'est un Nan

        X[self.col_select] = X.apply(
                lambda row: self.dict_sun.get((row[self.geo], row[self.col_target]), row[self.col_select])
                if pd.isna(row[self.col_select]) else row[self.col_select],
                axis=1)

        return X
