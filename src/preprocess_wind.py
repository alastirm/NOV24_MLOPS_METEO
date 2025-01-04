import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# prérequis à l'utilisation de la fonction
# la colonne climate(minuscule) doit être ajoutée au df
# les na de la colonne RainTomorrow doivent être supprimer


# df = pd.read_csv('../data/weatherAUS.csv')
# df['climate'] = df['Location'].map(climate)
# df = df.dropna(subset = 'RainTomorrow')

# Bruno : je propose de changer en Climate avec C majuscule

class wind_speed_transformer(BaseEstimator, TransformerMixin):
    '''
    cette classe permet de gérer les colonnes vents quantitatives.
    elle prend en argument :
    geo : climate, mais peut être changer par location,
    col_select : windgustspeed, colonne quantitative sur laquelle on veut remplacer les Nan,
    col_target : RainTomorrow
    method : median, peut être remplacer par mean, mod, etc
    '''


    def __init__(self, geo :str = 'climate', col_select : str = 'WindGustSpeed', col_target : str = 'RainTomorrow', method : str = 'median'):
        self.dict_wgs = {}
        self.geo = geo
        self.col_select = col_select
        self.col_target = col_target
        self.method = method

    def fit(self, X, y = None):
        # on récupère dans un dictionnaire la valeur de method pour la col_select dont on veut gérer les Nan, avec clé tuple(geo, col_target)

        for i in X[self.geo].unique():
            for j in X[self.col_target].unique():
                self.dict_wgs[i, j] = getattr(X[(X[self.geo] == i) & (X[self.col_target] == j)][self.col_select], self.method)()

        return self

    def transform(self, X):
        # on cherche le tuple(geo, col_target) et on applique la valeur de la method si c'est un Nan

        X[self.col_select] = X.apply(
                lambda row: self.dict_wgs.get((row[self.geo], row[self.col_target]), row[self.col_select])
                if pd.isna(row[self.col_select]) else row[self.col_select],
                axis=1)

        return X
