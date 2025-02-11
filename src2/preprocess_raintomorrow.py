import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class raintomorrow_transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X['RainTomorrow'] = X['RainTomorrow'].replace(['Yes', 'No'], [1, 0]).astype(float)
        X.dropna(subset = 'RainTomorrow', inplace = True)

        X['lag_raintomorrow'] = X['RainTomorrow'].shift()

        X['RainTomorrow' +'_mean_7'] = X['RainTomorrow'].rolling(window=7, min_periods=1).mean().dropna()
        X['RainTomorrow' + '_std_7'] = X['RainTomorrow'].rolling(window=7, min_periods=1).std().dropna()
        X['RainTomorrow' + '_max_7'] = X['RainTomorrow'].rolling(window=7, min_periods=1).max().dropna()
        X['RainTomorrow' + '_min_7'] = X['RainTomorrow'].rolling(window=7, min_periods=1).min().dropna()

        X['RainTomorrow' +'_mean_5'] = X['RainTomorrow'].rolling(window=5, min_periods=1).mean().dropna()
        X['RainTomorrow' + '_std_5'] = X['RainTomorrow'].rolling(window=5, min_periods=1).std().dropna()
        X['RainTomorrow' + '_max_5'] = X['RainTomorrow'].rolling(window=5, min_periods=1).max().dropna()
        X['RainTomorrow' + '_min_5'] = X['RainTomorrow'].rolling(window=5, min_periods=1).min().dropna()

        X['RainTomorrow' +'_mean_3'] = X['RainTomorrow'].rolling(window=3, min_periods=1).mean().dropna()
        X['RainTomorrow' + '_std_3'] = X['RainTomorrow'].rolling(window=3, min_periods=1).std().dropna()
        X['RainTomorrow' + '_max_3'] = X['RainTomorrow'].rolling(window=3, min_periods=1).max().dropna()
        X['RainTomorrow' + '_min_3'] = X['RainTomorrow'].rolling(window=3, min_periods=1).min().dropna()


        X.dropna(subset = 'RainTomorrow', inplace = True)

        return X
