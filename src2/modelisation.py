import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


def modelisation(df):

    X = df.drop(columns = 'RainTomorrow')
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    scaler = MinMaxScaler(feature_range = (-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poids = y_train.value_counts(normalize = True)


    model = LogisticRegression(verbose = 0,
                               C = 1,
                               max_iter = 1000,
                               penalty = 'l1',
                               solver = 'liblinear',
                               intercept_scaling = 0.1,
                               l1_ratio = 0.5,
                               tol = 0.01)

    model.fit(X_train_scaled, y_train)

    y_pred_prob1 = model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob1)
    roc_auc = roc_auc_score(y_test, y_pred_prob1)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (y_pred_prob1 > optimal_threshold).astype(int)


    print('score accuracy : ', model.score(X_test_scaled, y_test))
    print('f1 score : ', f1_score(y_test, y_pred))
    print('roc-auc score : ', roc_auc_score(y_test, y_pred))
    print('brier score : ', brier_score_loss(y_test, y_pred), '\n\n')

    print(confusion_matrix(y_test, y_pred), '\n\n')
    print(classification_report(y_test, y_pred))
