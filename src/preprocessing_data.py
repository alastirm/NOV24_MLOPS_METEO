import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

# import de la fonction initialize_data_weatherAU(data_dir)
import init_data

# Fonctions de preprocessing
import preprocess_RainTomorrow
import preprocess_Date
import preprocess_Rainfall_RainToday
import preprocess_wind

# Fonctions encodage
import encode_functions

# Autres fonctions ajoutées
import functions_created

# chargement des données
data_dir = "../data/weatherAUS.csv"
df = init_data.initialize_data_weatherAU(data_dir)

# Vérification chargement
df.head()
df.describe()

# print informations

nas_before_preprocess= pd.DataFrame(df.isna().sum())
print("Avant Preprocess : \n")
print("Nombre de Nas")
print(nas_before_preprocess)

dim_before_preprocess = df.shape
print("Dimensions : ", dim_before_preprocess)


# gestion des NAs et preprocessing des variables

# On supprime les colonnes Evaporation, Sunshine et Cloud
df = df.drop(columns=["Evaporation", 'Sunshine','Cloud9am','Cloud3pm'])

# preprocess Date
df = preprocess_Date.preprocess_Date(df)

# preprocess Variable cible
df = preprocess_RainTomorrow.preprocess_RainTomorrow(df)

# df_test = functions_created.complete_na_neareststation(df, variable="MinTemp", distance_max=50)

# preprocess Rainfall et RainToday

# Ce drop est indispensable pour que le transformer windDir fonctionne
df = df[(df['Location'] != 'Newcastle') & (df['Location'] != 'Albany')]
# preprocess wind
df = preprocess_wind.apply_transformer(df)

# dist_mat = functions_created.create_distance_matrix()

# affichage Nas après preprocessing

nas_after_preprocess = pd.DataFrame(df.isna().sum())
nas_after_preprocess = pd.merge(nas_before_preprocess,
                                nas_after_preprocess,
                                left_index=True,
                                right_index=True)

nas_after_preprocess.columns=['Avant','Après']

print("Après Preprocess : \n")
print("Nombre de Nas")
print(nas_after_preprocess)

dim_after_preprocess = df.shape
print("Dimensions avant : ", dim_before_preprocess)
print("Dimensions après : ", dim_after_preprocess)

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

# Encodage des features
# Exemple en important la fonction encode_data de encode_functions.py
# On crée des dummies pour la saison, l'année, le mois avec le OneHotEncoder

vars_to_encode = ["Season", "Year", "Month"] 
X_train.head()

X_train, X_test = encode_functions.encode_data(X_train = X_train,
                                               X_test=X_test,
                                               vars_to_encode=vars_to_encode,
                                               encoder="OneHotEncoder")

# Scaling
# Normalisation
# Choix des colonnes à scaler
vars_to_scale  = ['Rainfall']

# On fit sur Xtrain
scaler = MinMaxScaler().fit(X_train[vars_to_scale])

X_train_scaled = X_train 
X_train_scaled[vars_to_scale] = scaler.transform(X_train[vars_to_scale])
X_test_scaled = X_test
X_test_scaled[vars_to_scale] = scaler.transform(X_test[vars_to_scale])


print(X_train_scaled[vars_to_scale].describe())
print(X_test_scaled[vars_to_scale].describe())

# Sauvegarde fin preprocessing
X_train_scaled.to_csv("../data_saved/X_train_final.csv")
X_test_scaled.to_csv("../data_saved/X_test_final.csv")
y_train.to_csv("../data_saved/y_train_final.csv")
y_test.to_csv("../data_saved/y_test_final.csv")
