import sys
import os

import streamlit as st
import pandas as pd
import numpy as np

# modèles 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.svm import LinearSVC

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report,accuracy_score, f1_score, fbeta_score,  average_precision_score
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

# scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# import de modèle entraîné

import pickle
# gestion des chemins 
from pathlib import Path
sys.path.insert(0, './src/')

cwd = os.getcwd()
# import des fonctions de modélisations
import modeling_functions as mf

# classifier
classifier_list = {
    'LogisticRegression': LogisticRegression(
        class_weight={0: 0.3, 1: 0.7},
        C = 0.1,
        max_iter = 500, 
        penalty='l1', 
        solver = 'liblinear',
        n_jobs=-1),
    'RandomForestClassifier': RandomForestClassifier(
        class_weight={0: 0.3, 1: 0.7}, 
        criterion='log_loss',
        max_depth=10, 
        n_estimators=50,
        n_jobs=-1),
    'BalancedRandomForestClassifier': BalancedRandomForestClassifier(
        class_weight={0: 0.1, 1: 0.9},
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
        class_weight={0: 0.3, 1: 0.7},
        penalty='l1')
}

# métriques de grid_search
grid_metric_list = {"accuracy" : make_scorer(accuracy_score),
                "average_precision" : make_scorer(average_precision_score),
                "f1_score" : make_scorer(f1_score), 
                "f05_score" : make_scorer(fbeta_score, beta=0.5, pos_label=1),
                "f2_score" : make_scorer(fbeta_score, beta=2, pos_label=1)}


# jeu de données
dataset_list = ["test","V2t","V3t"]
dataset_list = ["test", "V2 : variables initiales","V3 : variables composées (variation, moyenne glissante)"]

threshold = 0.25

# initialisation booléens

model_loaded = False

# début stream lit
st.title("Modélisation sur l'ensemble des données")

dataset_choice = st.selectbox('Choisissez votre jeu de données', dataset_list, index = None)

if dataset_choice == "test":
    df =  pd.read_csv("./data_saved/data_preprocessed_V3.csv", 
                      index_col=["id_Location","id_Date"])
    dataset = "test"

if dataset_choice == "V2 : variables initiales":
    st.write("WARNING :  La multicolinéarité n'est pas traitée dans ces données")
    df =  pd.read_csv("./data_saved/data_preprocessed_V2.csv", 
                      index_col=["id_Location","id_Date"])
    dataset = "V2t"

elif dataset_choice == "V3 : variables composées (variation, moyenne glissante)":
    df =  pd.read_csv("./data_saved/data_preprocessed_V3.csv", 
                      index_col=["id_Location","id_Date"])
    dataset = "V3t"

if dataset_choice :
    missing_percentages = df.isna().mean()
    # Colonnes à conserver
    columns_to_keep = missing_percentages[missing_percentages <= threshold].index
    columns_dropped = missing_percentages[missing_percentages > threshold].index
    df = df[columns_to_keep]
    st.write("Dimensions du dataframe :", df.dropna().shape)
    st.write("Colonnes supprimées sur le dataframe :", columns_dropped.values)

    # séparation et scaling data
    X_train, X_test, y_train, y_test = \
    mf.separation_train_test(df, sep_method="temporelle", split = 0.8)
    X_train, X_test = mf.scaling(X_train, X_test, scaler = MinMaxScaler())

    # Choix du classificateur
    clf_choice = st.selectbox('Choisissez votre classificateur', classifier_list.keys(), 
                              index=None)

    st.write('Le modèle choisi est :', clf_choice)

    optim_choice = st.checkbox('Paramètres optimisés')

    if optim_choice: 
        grid_metric_choice = st.selectbox('Choisissez la métrique à optimiser ',
                                          grid_metric_list, index=None)
        
        modeling_batch = "simplegrid"
        
        if grid_metric_choice:
            # Chargement du modèle correspondant
            model_dir = "./saved_models/global/" + dataset + "/" + modeling_batch + "/"
             
            check_model_exist = Path(model_dir + grid_metric_choice + clf_choice + "search.pkl").exists()
            
            if check_model_exist:
                st.write("chargement du modèle dans le dossier :", model_dir + grid_metric_choice + clf_choice + "search.pkl")

                # charge le gridsearch
                #with open(model_dir  + grid_metric_choice  + clf_choice  +"search.pkl", 'rb') as s:
                #    search = pickle.load(s)

                # charge le modèle
                with open(cwd+ model_dir  + grid_metric_choice + "_" + clf_choice +"_tuned.pkl", 'rb') as m:
                    model = pickle.load(m)


                st.write("Paramètres du modèle : \n ", model)
                model_loaded = True
            
            else: 
                st.write("Absence du modèle dans le dossier :", model_dir + grid_metric_choice + clf_choice + "search.pkl")
                parameters_choice = st.selectbox('Choisissez vos paramètres ', 'par défaut', index = None)
                model = classifier_list[clf_choice]
                if parameters_choice:
                   st.write("Le modèle va être entraîné avec les paramètres suivants : \n ", model) 
                   model.fit(X_train, y_train)
                   model_loaded = True


    if model_loaded:
        # prédictions

        y_pred = model.predict(X_test)
        

        st.write("Choisissez les résultats à afficher")

        inter_col, cm_col, report_col = st.columns(3, border = True)

        with inter_col : 
            inter_choice = st.checkbox("Graphique d'interprétabilité")
        with cm_col : 
            cm_choice = st.checkbox("Matrice de confusion")
        with report_col : 
            report_choice = st.checkbox("Métriques")

        if inter_choice:
            n_feat = st.slider('Nombres de features', 
                               min_value = 1, 
                               max_value = len(X_train.columns), 
                               value = 20)
        else:
            n_feat=20

        if inter_choice or cm_choice or report_choice:

            report, cm, plot_res = mf.st_plot_model_results(
                model_name = clf_choice, model= model,
                X_train = X_train, X_test = X_test, 
                y_train = y_train, y_test =y_test,
                plot_inter = inter_choice,
                plot_cm = cm_choice,
                plot_report = report_choice,
                n_coef_graph=n_feat,
                )    
            # st.dataframe(report)
            st.pyplot(plot_res)

        # proba
        # y_probs = model.predict_proba(X_test)

        st.write("Choisissez un approfondissement")
        seuil_col, location_col = st.columns(2, border = True)
        with seuil_col: 
            seuil_choice = st.checkbox("Seuil de classification")
        with location_col: 
            location_choice = st.checkbox("Qualité des prédictions par station")

        if seuil_col:
            seuil_plot, metric_plot = \
                mf.st_plot_seuil_roc_AUC(
                    model,
                    X_train, X_test, y_train, y_test
                            )
            
            st.pyplot(seuil_plot)
            st.pyplot(metric_plot)

    else:
        print ("Impossible de charger ce modèle")


