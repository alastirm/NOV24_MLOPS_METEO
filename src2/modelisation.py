import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, precision_recall_curve, brier_score_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


def modelisation(df):

    X = df.drop(columns = 'RainTomorrow')
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    scaler = MinMaxScaler(feature_range = (-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poids = y_train.value_counts(normalize = True)


    model = RandomForestClassifier(
                n_estimators = 755,
                max_depth = 8,
                max_features = 'sqrt',
                min_samples_split = 3,
                min_samples_leaf = 15,
                criterion = 'gini',
                bootstrap = True,
                class_weight={0: 1/poids[0], 1: 1/poids[1]},
                random_state=42,
            )

    model.fit(X_train_scaled, y_train)

    print(y_train.value_counts(normalize = True))
    print(y_train.value_counts())

    prob = model.predict_proba(X_test_scaled)

    y_pred_prob1 = model.predict_proba(X_test_scaled)[:, 1]

    if y_train.value_counts(normalize = True)[1] < 0.15:
            threshold = 0.165
    if y_train.value_counts(normalize = True)[1] > 0.30:
        threshold = 0.85
    else:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob1)
        diff = abs(precision - recall)
        threshold = thresholds[diff.argmin()]
    print(threshold)


    y_pred = (y_pred_prob1 > threshold).astype(int)


    print('score accuracy : ', model.score(X_test_scaled, y_test))
    print('f1 score : ', f1_score(y_test, y_pred))
    print('roc-auc score : ', roc_auc_score(y_test, y_pred))
    print('brier score : ', brier_score_loss(y_test, y_pred), '\n\n')

    print(confusion_matrix(y_test, y_pred), '\n\n')
    print(classification_report(y_test, y_pred))
