import pandas as pd
import numpy as np
import calendar

from sklearn.preprocessing import LabelEncoder


class preprocess_date_transformer():

    def __init__(self):

        self.le = LabelEncoder()

    def fit(self, X, y = None):

        X['Date'] = pd.to_datetime(X['Date'])
        self.le.fit(X['Date'].dt.year)

        return self

    def transform(self, X):

        X['year'] = X['Date'].dt.year
        X['months'] = X['Date'].dt.month
        # X['day'] = X['Date'].dt.day

        X['year'] = self.le.transform(X['year'])

        X['sin_months'] = X['months'].apply(lambda x : np.sin((2 * np.pi) * (( x - 1 ) / 12)))
        X['cos_months'] = X['months'].apply(lambda x : np.cos((2 * np.pi) * (( x - 1 ) / 12)))
        # X['sin_day'] = X.apply(lambda row: np.sin((2 * np.pi) * ((row['day'] - 1) / calendar.monthrange(row['year'], row['months'])[1])), axis=1)
        # X['cos_day'] = X.apply(lambda row: np.cos((2 * np.pi) * ((row['day'] - 1) / calendar.monthrange(row['year'], row['months'])[1])), axis=1)

        X.drop(columns = ['months'], inplace = True)

        return X
