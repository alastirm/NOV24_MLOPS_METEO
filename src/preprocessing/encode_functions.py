import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import emoji


# Cette fonction encore les variables vars_to_encode avec l'encodeur OneHot
# en faisant le fit sur le X_train et le transform sur Xtrain et X_test
# Elle retourne X_train et X_test en sortie

def encode_data(X_train, X_test, vars_to_encode, encoder="OneHotEncoder"):

    # Encodage des variables
    if encoder ==  "OneHotEncoder":
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,  dtype='int')
    
    # Encodage des variables
    if encoder ==  "OrdinalEncoder":
        encoder = OrdinalEncoder()

    # On fit l'encodeur sur les variables voulues sur le jeu d'entrainement
    
    encoder.fit(X_train[vars_to_encode])

    # On crée les dummies
    var_enc_train = encoder.transform(X_train[vars_to_encode])
    var_enc_train = pd.DataFrame(var_enc_train,
                                 columns=encoder.get_feature_names_out(), 
                                 index=X_train.index)
    
    # On crée les dummies
    var_enc_test = encoder.transform(X_test[vars_to_encode])
    var_enc_test = pd.DataFrame(var_enc_test,
                                columns= encoder.get_feature_names_out(), 
                                index = X_test.index)


    # On ajoute les variables encodées au X_train de base et on supprime les variables d'origine
    X_train = pd.merge(X_train, var_enc_train,
                       left_index=True, right_index=True)
    
    X_train = X_train.drop(columns=vars_to_encode)

    # On ajoute les variables encodées au X_test de base et on supprime les variables d'origine
    X_test = pd.merge(X_test, var_enc_test,
                      left_index=True, right_index=True)
    
    X_test = X_test.drop(columns=vars_to_encode)

    return X_train, X_test

# Variante avec une classe, un choix de l'encoder (pas fonctionnel pour le moment hormis onehot), et une liste de variable
class encoder_vars(BaseEstimator, TransformerMixin):

    def __init__(self, vars_to_encode, encoder="OneHotEncoder"):
        # Encodage des variables
        self.vars_to_encode = vars_to_encode
        self.encoder = encoder
        if self.encoder == "OneHotEncoder":
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,  dtype='int')
        if encoder ==  "OrdinalEncoder":
            self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        # On fit l'encodeur sur les variables voulues
        self.encoder.fit(X[self.vars_to_encode])
        return self


    def transform(self, X):
        # On crée les variables encodées 
        var_enc = self.encoder.transform(X[self.vars_to_encode])
        var_enc = pd.DataFrame(var_enc,
                               columns=self.encoder.get_feature_names_out(), 
                               index=X.index)
        # On ajoute les variables encodées au X_train de base et on supprime les variables d'origine
        X = pd.merge(X, var_enc,
                     left_index=True, right_index=True)
    
        X = X.drop(columns=self.vars_to_encode)
        for col in self.vars_to_encode:
            print(f"encodage {col} {emoji.emojize(':thumbs_up:')}")

        return X
class trigo_encoder(BaseEstimator, TransformerMixin):

    def __init__(self, col_select):
        self.col_select = col_select
        self.mapping_cos_sin = {}


    def fit(self, X, y=None):
        angle_shift = 360/len(X[self.col_select].unique())
        mapping = {}
        i = 0
        for elt in X[self.col_select].unique():
            mapping[elt] = i*angle_shift
            i += 1

        for elt, angle in mapping.items():
            cos_a = np.round(np.cos(np.radians(angle)), 6)
            sin_a = np.round(np.sin(np.radians(angle)), 6)

            self.mapping_cos_sin[elt] = (cos_a, sin_a)

        return self


    def transform(self, X):

        X[self.col_select + '_cos'] = X[self.col_select].apply(lambda x : float(self.mapping_cos_sin[x][0]))
        X[self.col_select + '_sin'] = X[self.col_select].apply(lambda x : float(self.mapping_cos_sin[x][1]))

        print(f"encodage {self.col_select} {emoji.emojize(':thumbs_up:')}")

        return X


