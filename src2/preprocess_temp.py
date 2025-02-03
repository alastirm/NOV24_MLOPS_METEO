import pandas

class preprocess_temp():

    def __init__(self, col_select : str = 'MinTemp', method : str = 'linear'):

        self.col_select = col_select
        self.method = method

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        X[self.col_select] = X[self.col_select].interpolate(method = self.method)

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
