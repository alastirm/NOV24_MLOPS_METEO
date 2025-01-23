import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def modelisation(df):

    X = df.drop(columns = 'RainTomorrow')
    y = df['RainTomorrow']

    cut_index = df.index[int(0.8 * len(df))]  # 80% des données pour l'entraînement

    # Diviser les données en fonction de la date
    X_train = X[df.index <= cut_index]
    y_train = y[df.index <= cut_index]
    X_test = X[df.index > cut_index]
    y_test = y[df.index > cut_index]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators = 1000, class_weight={0: 0.25, 1: 0.75})
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print('score accuracy : ', model.score(X_test_scaled, y_test), end = '\n\n')
    print('f1 score : ', f1_score(y_test, y_pred))
    print('roc-auc score : ', roc_auc_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# print
