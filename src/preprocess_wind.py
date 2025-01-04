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

    def __init__(self, geo :str = 'climate', col_select : str = 'WindGustSpeed', col_target : str = 'RainTomorrow', method : str = 'median'):
        self.dict_wgs = {}
        self.geo = geo
        self.col_select = col_select
        self.col_target = col_target
        self.method = method

    def fit(self, X, y = None):
        # on récupère dans un dictionnaire la valeur median Windgustspeed des paires climate, raintomorrow

        for i in X[self.geo].unique():
            for j in X[self.col_target].unique():
                self.dict_wgs[i, j] = getattr(X[(X[self.geo] == i) & (X[self.col_target] == j)][self.col_select], self.method)()

        return self

    def transform(self, X):
        # on applique la valeur mediane à la paire climate, raintomorrow si elle est manquante

        X[self.col_select] = X.apply(
                lambda row: self.dict_wgs.get((row[self.geo], row[self.col_target]), row[self.col_select])
                if pd.isna(row[self.col_select]) else row[self.col_select],
                axis=1)

        return X
