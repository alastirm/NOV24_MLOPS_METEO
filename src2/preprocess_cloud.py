import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class cloud_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_select: str = 'Cloud9am', col_target: str = 'RainTomorrow', method: str = 'median'):

        self.dict_cld = {}
        self.col_select = col_select
        self.col_target = col_target
        self.method = method


    def fit(self, X, y=None):

        self.dict_cld = X.groupby(self.col_target)[self.col_select].agg(self.method).to_dict()

        return self


    def transform(self, X):

        X[self.col_select] = X.apply(
            lambda row: self.dict_cld.get(row[self.col_target], row[self.col_select])
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
