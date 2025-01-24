import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score



def modelisation(df):

    X = df.drop(columns = 'RainTomorrow')
    y = df['RainTomorrow']

    print('demande....', y.value_counts(normalize = True))

    # cut_index = df.index[int(0.8 * len(df))]
    # X_train = X[df.index <= cut_index]
    # y_train = y[df.index <= cut_index]
    # X_test = X[df.index > cut_index]
    # y_test = y[df.index > cut_index]



    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    scaler = MinMaxScaler(feature_range = (-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter = 10000,
                               verbose = 1,
                               class_weight = {0 : 0.75, 1 : 0.25})


    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print('score accuracy : ', model.score(X_test_scaled, y_test), end = '\n\n')
    print('f1 score : ', f1_score(y_test, y_pred))
    print('roc-auc score : ', roc_auc_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# print
