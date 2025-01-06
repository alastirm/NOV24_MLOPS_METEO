import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data

# Fonctions de preprocessing
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_Rainfall_RainToday  
import preprocess_wind  

# Autres fonctions ajoutées
from functions_created import create_date_list

# chargement des données
data_dir = "../data_csv/weatherAUS_091224.csv"
df = init_data.initialize_data_weatherAU(data_dir)
 
# Vérification chargement
df.head()
df.describe()

# print informations

print("Avant Preprocess : \n")
print("Nombre de Nas")
print(df.isna().sum())

print("Dimensions : ", df.shape)

# gestion des NAs et preprocessing des variables

# On supprime les colonnes Evaporation et Sunshine
df = df.drop(columns = ["Evaporation", 'Sunshine'])

# preprocess Date
df = preprocess_Date.preprocess_Date(df)

# preprocess Variable cible
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)

# preprocess Rainfall et RainToday
df = preprocess_Rainfall_RainToday.preprocess_Rainfall_RainToday(df)

# preprocess wind (ne fonctionne plus pour le moment)
# df = preprocess_wind.preproc_windgustspeed(df)


print("après Preprocess : \n")
print("Nombre de Nas")
print(df.isna().sum())

print("Dimensions : ", df.shape)

# On retire les derniers Nas (à faire après avoir géré toutes les colonnes)
df_final = df.dropna()

# On retire les colonnes Date et Location qui sont en index
df_final = df_final.drop(columns=["Date", "Location"])

# On sélectionne les features à garder

feats_selected = ['Year', 'Month', 'Season',
                  'RainToday', 'Rainfall']

# Scindage du dataset en un échantillon test (20%) et train

feats = df_final.drop(columns="RainTomorrow")
feats = feats.loc[:,feats_selected]
target = df_final["RainTomorrow"]
X_train, X_test, y_train, y_test = \
    train_test_split(feats, target, test_size=0.20, random_state=1234)

# Encodage 
# Exemple en important les fonctions de preprocess_Date.py 

X_train = preprocess_Date.encode_Date(X_train)
X_test = preprocess_Date.encode_Date(X_test)


# Scaling
# Normalisation
# Choix des colonnes à scaler
cols_to_scale  = ['Rainfall']

# On fit sur Xtrain
scaler = MinMaxScaler().fit(X_train[cols_to_scale])

X_train_scaled = X_train 
X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test_scaled = X_test
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


print(X_train_scaled[cols_to_scale].describe())
print(X_test_scaled[cols_to_scale].describe())

# Gestion du déséquilibre et resampling (A discuter)

# On utilise le rééchantillonnage pour traiter le déséquilibre de la variable cible -> SMOTE
from imblearn.over_sampling import SMOTE

print('Classes échantillon initial :', dict(pd.Series(target).value_counts()))

smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))


# Fin du preprocessing : on peut faire un rapide feature selection

from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

k_select = len(X_sm.columns)

# On garde la moitié des variables environ avec un scoring de fisher
selector = SelectKBest(f_classif, k=20)
selector = selector.fit(X_sm, y_sm)
selector.get_feature_names_out()

X_sm = selector.transform(X_sm)
X_test = selector.transform(X_test)


# test un modèle basique
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

lr = LogisticRegression(max_iter = 1000)
lr.fit(X_sm, y_sm, )
y_pred = lr.predict(X_test)
print(pd.crosstab(y_test, y_pred))
print(classification_report_imbalanced(y_test, y_pred))