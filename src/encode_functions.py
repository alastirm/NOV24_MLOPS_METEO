import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Cette fonction encore les variables vars_to_encode avec l'encodeur OneHot
# en faisant le fit sur le X_train et le transform sur Xtrain et X_test
# Elle retourne X_train et X_test en sortie

def encode_data(X_train, X_test, vars_to_encode, encoder="OneHotEncoder"):

    # Encodage des variables
    if encoder ==  "OneHotEncoder":
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,  dtype='int')
    
    # Saison : dummies par saison
    # Mois : dummies par saison
    # Year : dummies par saison

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


