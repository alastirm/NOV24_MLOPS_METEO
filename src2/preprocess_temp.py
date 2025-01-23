import pandas

class preprocess_temp():

    def __init__(self, col_select : str = 'MinTemp', method : str = 'linear'):

        self.col = col_select
        self.method = method

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        X[self.col] = X[self.col].interpolate(method = self.method)

        return X
