import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# features selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# modèles 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

# Rééchantillonage
from imblearn.over_sampling import SMOTE

# Autres fonctions
import functions_created

# Ce script balaye quelques méthodes de features selection sur un jeu de données nettoyé

# Chargement données issues du preprocessing

X_train = pd.read_csv("../data_saved/X_train_final.csv", index_col=["id_Location","id_Date"])
X_test = pd.read_csv("../data_saved/X_test_final.csv", index_col=["id_Location","id_Date"])
y_train = pd.read_csv("../data_saved/y_train_final.csv", index_col=["id_Location","id_Date"])
y_test = pd.read_csv("../data_saved/y_test_final.csv", index_col=["id_Location","id_Date"]) 

X_train.head()
X_test.head()
y_train.head()
y_test.head()

# sauvegarde avant sélection
X_train_save = X_train
X_test_save = X_test

# Gestion du déséquilibre et resampling (A discuter)

# On utilise le rééchantillonnage pour traiter le déséquilibre de la variable cible -> SMOTE

print('Classes échantillon initial :', dict(pd.Series(y_train["RainTomorrow"]).value_counts()))

smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm["RainTomorrow"]).value_counts()))

# on peut faire un rapide feature selection ou utiliser le script features_selection.py pour aller plus loin

k_select = len(X_sm.columns)

# On garde la moitié des variables environ avec un scoring de fisher
selector = SelectKBest(f_classif, k=20)
selector = selector.fit(X_sm, y_sm)
selector.get_feature_names_out()

X_sm = selector.transform(X_sm)
X_test = selector.transform(X_test)

# test un modèle basique

lr = LogisticRegression(max_iter = 1000)
lr.fit(X_sm, y_sm, )
y_pred = lr.predict(X_test)
print(pd.crosstab(y_test["RainTomorrow"], y_pred))
print(classification_report_imbalanced(y_test["RainTomorrow"], y_pred))