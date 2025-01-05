import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed



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

    def transform(self, X):
        # choisit la valeur dans dict_wdt si col_select est Nan et en fonction de la valeur de col_target
        # sinon conserve la valeur de col_select

        X[self.col_select] = X.apply(
            lambda row: self.dict_wdt[row[self.geo]][0] if pd.isna(row[self.col_select]) and row[self.col_target] == 0
                  else self.dict_wdt[row[self.geo]][1] if pd.isna(row[self.col_select]) and row[self.col_target] == 1
                  else row[self.col_select],
            axis=1)

        return X


def apply_transformer(df):
    # cette fonction final a besoin d'être finalisé car elle renvoie une liste avec 6 dataframe

    transformers = [
        wind_speed_transformer(col_select='WindGustSpeed'),
        wind_speed_transformer(col_select='WindSpeed9am'),
        wind_speed_transformer(col_select='WindSpeed3pm'),
        wind_dir_transformer(col_select='WindGustDir'),
        wind_dir_transformer(col_select='WindDir9am'),
        wind_dir_transformer(col_select='WindDir3pm')
    ]

    df_transformed = Parallel(n_jobs=-1)(delayed(apply_transformer)(t, df) for t in transformers)

    return df_transformed
