import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# search hyperparameters
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV

# modèles 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.svm import LinearSVC

import xgboost as xgb
import shap

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report,accuracy_score, f1_score, fbeta_score,  average_precision_score
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, precision_recall_curve

# scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# sauvegarde model
import pickle
import itertools
import os

sys.path.insert(0, './src/modeling')
# import des fonctions de modélisations
import modeling_functions as mf

# jeu de données issues du preprocessing successif
df_V2 =  pd.read_csv("../../data_saved/data_preprocessed_V2.csv", index_col=["id_Location","id_Date"])
missing_percentages = df_V2.isna().mean()
# Colonnes à conserver
threshold = 0.25
columns_to_keep = missing_percentages[missing_percentages <= threshold].index
columns_dropped = missing_percentages[missing_percentages > threshold].index
df_V2 = df_V2[columns_to_keep]
print("Colonnes supprimées sur le df V2 :", columns_dropped)
print(missing_percentages.loc["Evaporation"])

# séparation test/train
X_train_V2t, X_test_V2t, y_train_V2t, y_test_V2t = \
    mf.separation_train_test(df_V2, sep_method="temporelle", split = 0.8)

# scaling 

X_train_V2t_scaled, X_test_V2t_scaled = mf.scaling(X_train_V2t, X_test_V2t, scaler = MinMaxScaler())

# Choix de quelques modèles 
models_select  = {
    'LogisticRegression': LogisticRegression(
#        class_weight={0: 0.3, 1: 0.7},
        C = 1,
        max_iter = 500, 
        penalty='l1', 
        solver = 'liblinear',
        n_jobs=-1),
    'RandomForestClassifier': RandomForestClassifier(
#        class_weight={0: 0.3, 1: 0.7}, 
        criterion='log_loss',
        max_depth=10,
        n_estimators=50,
        n_jobs=-1),
    'BalancedRandomForestClassifier': BalancedRandomForestClassifier(
#        class_weight={0: 0.1, 1: 0.9},
        criterion='entropy', 
        max_depth=30,
        n_estimators=200,
        n_jobs=-1),
    'BaggingClassifier': BaggingClassifier(
        max_features=1.0,
        max_samples=0.5,
        n_estimators=1000,
         n_jobs=-1),
    'BalancedBaggingClassifier': BalancedBaggingClassifier(
        max_features=1.0,
        max_samples=0.5,
        n_estimators=1000,
         n_jobs=-1),
    'LinearSVC' : LinearSVC(
        max_iter=500, 
#        class_weight={0: 0.3, 1: 0.7},
        penalty='l1')                            
}

# fit et sauvegarde sur données preprocessed V2

report_V2t, cm_V2t, models_V2t = \
    mf.fit_models(models_select, 
               X_train_V2t_scaled, X_test_V2t_scaled, y_train_V2t, y_test_V2t,
               save_model = True, save_models_dir="base_V2t_",
               save_results=True,save_results_dir="Base/base_V2t_")
