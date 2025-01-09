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
from sklearn.metrics import classification_report

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

# test un modèle de régression logistique sur tout le jeu de données

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(pd.crosstab(y_test["RainTomorrow"], y_pred))
print(classification_report(y_test["RainTomorrow"], y_pred))

# Gestion du déséquilibre et resampling (A discuter)
# On utilise le rééchantillonnage pour traiter le déséquilibre de la variable cible -> SMOTE

print('Classes échantillon initial :', dict(pd.Series(y_train["RainTomorrow"]).value_counts()))

smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm["RainTomorrow"]).value_counts()))

# test un modèle de régression logistique sur le jeu de données resample

lr_sm = LogisticRegression(max_iter=1000)
lr_sm.fit(X_sm, y_sm)
y_pred_sm = lr_sm.predict(X_test)
print(pd.crosstab(y_test["RainTomorrow"], y_pred_sm))
print(classification_report_imbalanced(y_test["RainTomorrow"], y_pred_sm))

# on peut aussi faire un rapide feature selection ou utiliser le script features_selection.py pour aller plus loin

k_select = len(X_sm.columns)

# On garde la moitié des variables environ avec un scoring de fisher

selector = SelectKBest(f_classif, k=k_select)
selector = selector.fit(X_sm, y_sm)
selector.get_feature_names_out()

X_sm_sel = selector.transform(X_sm)
X_test_sel = selector.transform(X_test)

# test un modèle de régression logistique sur le jeu de données resample avec moins de features

lr_sel = LogisticRegression(max_iter=1000)
lr_sel.fit(X_sm_sel, y_sm, )
y_pred_sel = lr_sel.predict(X_test_sel)
print(pd.crosstab(y_test["RainTomorrow"], y_pred_sel))
print(classification_report_imbalanced(y_test["RainTomorrow"], y_pred_sel))


# Teste un modèle sur une location

location = "Sydney"
X_train_loc = X_train.loc[location]
y_train_loc = y_train.loc[location]
X_test_loc = X_test.loc[location]
y_test_loc = y_test.loc[location]

lr_loc = LogisticRegression(max_iter=1000)
lr_loc.fit(X_train_loc, y_train_loc)
y_pred_loc = lr.predict(X_test_loc)
print(pd.crosstab(y_test_loc["RainTomorrow"], y_pred_loc))
print(classification_report(y_test_loc["RainTomorrow"], y_pred_loc))


print('Classes échantillon initial :',
      dict(pd.Series(y_train_loc["RainTomorrow"]).value_counts()))

smo = SMOTE()
X_sm_loc, y_sm_loc = smo.fit_resample(X_train_loc, y_train_loc)
print('Classes échantillon SMOTE :',
      dict(pd.Series(y_sm_loc["RainTomorrow"]).value_counts()))

lr_loc = LogisticRegression(max_iter=1000)
lr_loc.fit(X_sm_loc, y_sm_loc)
y_pred_loc = lr.predict(X_test_loc)
print(pd.crosstab(y_test_loc["RainTomorrow"], y_pred_loc))
print(classification_report_imbalanced(y_test_loc["RainTomorrow"], y_pred_loc))
