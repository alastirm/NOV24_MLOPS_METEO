import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class raintomorrow_transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X['lag_raintomorrow'] = X['RainTomorrow'].shift()
        X['lag_raintomorrow'] = X['lag_raintomorrow'].replace(['Yes', 'No'], [1, 0])

        X['RainTomorrow'] = X['RainTomorrow'].replace(['Yes', 'No'], [1, 0])
        X.dropna(subset = 'RainTomorrow', inplace = True)

        return X
