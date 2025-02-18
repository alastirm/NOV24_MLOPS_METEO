import sys
import os

import emoji


import streamlit as st
import pandas as pd
import numpy as np
import datetime
import emoji

# modèles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.svm import LinearSVC

# métriques
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, fbeta_score,  average_precision_score
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

# page parameters
st.set_page_config(page_title = 'MeteoStralia',
                   layout = 'wide',
                   page_icon = emoji.emojize('🦘'))

# classifier
classifier_list = {
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

# métriques de grid_search
grid_metric_list = {"accuracy" : make_scorer(accuracy_score),
                "average_precision" : make_scorer(average_precision_score),
                "f1_score" : make_scorer(f1_score),
                "f05_score" : make_scorer(fbeta_score, beta=0.5, pos_label=1),
                "f2_score" : make_scorer(fbeta_score, beta=2, pos_label=1)}


# jeu de données
#dataset_list = ["V2t","V3t","V4t"]
dataset_list = ["V2 : Original dataset",
                "V3 : New variables (daily growth rate, rolling mean)",
                "V4 : New data addition",
                "V5 : New variables + New data"]

threshold = 0.25

# initialisation booléens

model1_loaded = False
model2_loaded = False
check_model_exist1 = False
check_model_exist2 = False

# début stream lit
#st.title("Model comparator on the whole dataset")
st.html("""
        <h1 style = "color : rgb(16, 0, 97); font-size : 200%;">Model comparator on the whole dataset</h1>
    </div>
""")
        
model1_col, model2_col = st.columns(2, border = True)

##############################
# Chargement premier modèle
##############################

with model1_col:

    st.header("Model 1")

    dataset_choice1 = st.selectbox('Select your dataset', dataset_list, index = None)

    if dataset_choice1 == "V2 : Original dataset":
        # st.write("WARNING :  La multicolinéarité n'est pas traitée dans ces données")
        df =  pd.read_csv("./data_saved/data_preprocessed_V2.csv",
                        index_col=["id_Location","id_Date"])
        dataset = "V2t"

    elif dataset_choice1 == "V3 : New variables (daily growth rate, rolling mean)":
        df =  pd.read_csv("./data_saved/data_preprocessed_V3.csv",
                        index_col=["id_Location","id_Date"])
        dataset = "V3t"

    elif dataset_choice1 == "V4 : New data addition":
        df =  pd.read_csv("./data_saved/data_preprocessed_V4.csv",
                        index_col=["id_Location","id_Date"])
        dataset = "V4t"

    elif dataset_choice1 == "V5 : New variables + New data":
        df =  pd.read_csv("./data_saved/data_preprocessed_V5.csv",
                        index_col=["id_Location","id_Date"])
        dataset = "V4t"

    if dataset_choice1 :
        missing_percentages = df.isna().mean()
        # Colonnes à conserver
        columns_to_keep = missing_percentages[missing_percentages <= threshold].index
        columns_dropped = missing_percentages[missing_percentages > threshold].index
        df = df[columns_to_keep]
        st.write("Dataframe shape :", df.dropna().shape)
        # st.write("Colonnes supprimées sur le dataframe :", columns_dropped.values)

        # séparation et scaling data
        X_train, X_test, y_train, y_test = \
        mf.separation_train_test(df, sep_method="temporelle", split = 0.8)
        X_train, X_test = mf.scaling(X_train, X_test, scaler = MinMaxScaler())

        # Choix du classificateur
        clf_choice1 = st.selectbox('Select your ML classifier', classifier_list.keys(), 
                                index=None)

        #st.write('Le modèle choisi est :', clf_choice1)

        # Choix entre chargement de paramètres optimisés ou réentrainement

        optim_col1, train_col1 = st.columns(2, border = True)

        with optim_col1:
            optim_choice1 = st.checkbox("Tuned parameters from gridsearch")

        with train_col1:
            train_choice1 = st.checkbox('New training')

        if optim_choice1: 
            grid_metric_choice1 = st.selectbox('Select the gridsearch scoring metric',
                                            grid_metric_list, index=None)

            modeling_batch = "simplegrid"

            if grid_metric_choice1:
                # Chargement du modèle correspondant
                model_dir = "./saved_models/global/" + dataset + "/" + modeling_batch + "/"

                check_model_exist1 = Path(model_dir + grid_metric_choice1 + clf_choice1 + "search.pkl").exists()

                if check_model_exist1:
                    st.write("Model loading from folder :", model_dir + grid_metric_choice1 + clf_choice1 + "search.pkl")

                    # charge le gridsearch
                    #with open(model_dir  + grid_metric_choice  + clf_choice  +"search.pkl", 'rb') as s:
                    #    search = pickle.load(s)

                    # charge le modèle
                    with open(cwd+ model_dir  + grid_metric_choice1 + "_" + clf_choice1 +"_tuned.pkl", 'rb') as m:
                        model1 = pickle.load(m)

                    st.write("Model Parameters : \n ", model1)
                    model1_loaded = True

                else:
                    st.write("No model at :", model_dir + grid_metric_choice1 + clf_choice1 + "search.pkl")
                    parameters_choice1 = st.selectbox('Select your parameters ', 'Default parameters', index = None)
                    model1 = classifier_list[clf_choice1]
                    if parameters_choice1:
                        st.write("The selected model will be trained with the following parameters : \n ", model1)
                        model1.fit(X_train, y_train)
                        model1_loaded = True

        elif train_choice1:
            parameters_choice1 = st.selectbox('Select your parameters ', 'Default parameters', index = None)
            model1 = classifier_list[clf_choice1]
            if parameters_choice1:
                modeling_batch = "livetrain"
                model_train_dir1 = "./saved_models/global/" + dataset + "/" + modeling_batch + "/"
                check_model_train_exist1 = Path(model_train_dir1 + clf_choice1 + ".pkl").exists()
                if check_model_train_exist1:
                    st.write("Model already exists. Loading it from folder :", model_train_dir1 + clf_choice1 + ".pkl")
                    with open(cwd+ model_train_dir1 +clf_choice1 +".pkl", 'rb') as m:
                        model1 = pickle.load(m)
                    model1_loaded = True
                else:
                    st.write("The selected model will be trained with the following parameters : \n ", model1) 
                    model1.fit(X_train, y_train)
                    model1_loaded = True

##############################
# Résultats premier modèle
##############################

    if model1_loaded:
            # prédictions

            y_pred = model1.predict(X_test)


            st.write("Select the results to print")

            inter_col, cm_col, report_col = st.columns(3, border = True)

            with inter_col :
                inter_choice = st.checkbox("Interpretability plot")
            with cm_col :
                cm_choice = st.checkbox("Confusion matrix")
            with report_col :
                report_choice = st.checkbox("Metrics")

            if inter_choice:
                n_feat = st.slider('Number of features',
                                min_value = 1,
                                max_value = len(X_train.columns),
                                value = 10)
            else:
                n_feat=10

            if inter_choice or cm_choice or report_choice:

                report, cm, plot_res = mf.st_plot_model_results(
                    model_name = clf_choice1, model= model1,
                    X_train = X_train, X_test = X_test,
                    y_train = y_train, y_test =y_test,
                    plot_inter = inter_choice,
                    plot_cm = cm_choice,
                    plot_report = report_choice,
                    n_coef_graph=n_feat,
                    )
                # st.dataframe(report)
                st.pyplot(plot_res)

                if train_choice1 or check_model_exist1==False:
                        st.write("Do you want to save this model with pickle?")
                    
                        save_model1 = st.checkbox("Save the model", key="m231")
                        # timestampnow = pd.Timestamp(datetime.datetime.now()).round(freq='min').timestamp()
                        # timestampnow = round(timestampnow,ndigits=0)
                        # pd.to_datetime(timestampnow, unit = "s")
                        if save_model1:
                            modeling_batch = "livetrain"
                            model_dir1 = "./saved_models/global/" + dataset + "/" + modeling_batch + "/"
                            os.makedirs(os.path.dirname(model_dir1), exist_ok=True)
                            with open(cwd + model_dir1  + clf_choice1 + ".pkl", 'wb') as m:
                                    pickle.dump(model1,m)

            # proba
            # y_probs = model.predict_proba(X_test)

                st.write("Select more criteria to deepen the results")
                seuil_col, location_col = st.columns(2, border = True)
                with seuil_col:
                    seuil_choice = st.checkbox("Classification treshold")
                with location_col:
                    location_choice = st.checkbox("Forecast by weather station")

                if location_choice:
                    plot_FN, plot_FP = mf.st_plot_FP_FN(
                        "Location",
                        X_train, X_test, y_train, y_test,
                        y_pred)
                    st.pyplot(plot_FN)
                    st.pyplot(plot_FP)

                if seuil_choice:
                    seuil_plot, metric_plot = mf.st_plot_seuil_roc_AUC(
                        model1,
                        X_train, X_test, y_train, y_test)

                    st.pyplot(seuil_plot)
                    st.pyplot(metric_plot)

                    threshold = st.slider("Treshold adjustment", 0.0, 1.0, value = 0.2, step=0.1)
                    adjust_report, cm_avant, cm_apres = mf.adjust_classification_treshold(model1,
                                        X_train, X_test, y_train, y_test,
                                        threshold)
                    col1, col2 = st.columns(2, border = False)
                    with col1:
                        st.dataframe(adjust_report.iloc[:,0])
                        st.dataframe(cm_avant)
                    with col2:
                        st.dataframe(adjust_report.iloc[:,1])
                        st.dataframe(cm_apres)

                st.write("Do you want to make a forecast?")
                location1, time1 = st.columns(2, border = True)
                with location1:
                    city = st.selectbox(label = 'Select a city',
                                options = sorted(df.index.get_level_values(0).unique()),
                                index = None,
                                label_visibility="visible",
                                help = 'get a listed city',
                                placeholder = 'No city selected yet'
                                )
                if city :
                    date1 = X_test.loc[city,:].index.get_level_values(0)[-10]
                    with time1:
                        date1 = st.date_input(label = 'Select a Date',
                                        help = 'year-month-day',
                                        value = date1)

                if city and date1:
                    datetomorrow1 = str(date1 + datetime.timedelta(days=1))
                    date1 = str(date1)
                    X_predict = np.array(X_test.loc[(city, date1),])
                    pred1 = model1.predict(X_predict.reshape(1, -1))
                    prob1 = model1.predict_proba(X_predict.reshape(1, -1))
                    if pred1 == 1:
                        st.write("Don't forget your umbrella, it is going to rain on the ",datetomorrow1, " at ", city)
                        st.write("Probability : ", str(round(prob1[0,1]*100, ndigits=0)), "%")
                    if pred1 == 0:
                        st.write("No rain on the ",datetomorrow1, " at ", city)
                        st.write("Probability : ", str(round(prob1[0,0]*100, ndigits=0)), "%")

                    y_true1 = y_test.loc[(city, date1),]
                    if  y_true1 == 1.0:
                        st.write("In reality, it rained at  ", city, " on the ", datetomorrow1)
                        st.write("It rained ", str(df.loc[(city, datetomorrow1),"Rainfall"]), " mm")
                    if  y_true1 == 0.0:
                        st.write("In reality, it rained at  ", city, " on the ", datetomorrow1)
                        st.write("It rained ", str(df.loc[(city, datetomorrow1),"Rainfall"]), " mm")

##############################
# Chargement deuxième modèle
##############################

with model2_col:

    st.header("Model 2")

    dataset_choice2 = st.selectbox('Select your dataset', dataset_list, index = None, key="m21")

    if dataset_choice2 == "V2 : Original dataset":
        # st.write("WARNING :  La multicolinéarité n'est pas traitée dans ces données")
        df2 =  pd.read_csv("./data_saved/data_preprocessed_V2.csv",
                        index_col=["id_Location","id_Date"])
        dataset2 = "V2t"

    elif dataset_choice2 == "V3 : New variables (daily growth rate, rolling mean)":
        df2 =  pd.read_csv("./data_saved/data_preprocessed_V3.csv",
                        index_col=["id_Location","id_Date"])
        dataset2 = "V3t"

    elif dataset_choice2 == "V4 : New data addition":
        df2 =  pd.read_csv("./data_saved/data_preprocessed_V4.csv",
                        index_col=["id_Location","id_Date"])
        dataset2 = "V4t"

    elif dataset_choice2 == "V5 : New variables + New data":
        df2 =  pd.read_csv("./data_saved/data_preprocessed_V5.csv",
                        index_col=["id_Location","id_Date"])
        dataset2 = "V5t"

    if dataset_choice2 :
        missing_percentages = df2.isna().mean()
        # Colonnes à conserver
        columns_to_keep = missing_percentages[missing_percentages <= threshold].index
        columns_dropped = missing_percentages[missing_percentages > threshold].index
        df2 = df2[columns_to_keep]
        st.write("Dataframe shape :", df2.dropna().shape)
        # st.write("Colonnes supprimées sur le dataframe :", columns_dropped.values)

        # séparation et scaling data
        X_train2, X_test2, y_train2, y_test2 = \
        mf.separation_train_test(df2, sep_method="temporelle", split = 0.8)
        X_train2, X_test2 = mf.scaling(X_train2, X_test2, scaler = MinMaxScaler())

        # Choix du classificateur
        clf_choice2 = st.selectbox('Select your ML classifier', classifier_list.keys(), 
                                index=None, key="m22")

        #st.write('Le modèle choisi est :', clf_choice1)

        # Choix entre chargement de paramètres optimisés ou réentrainement

        optim_col2, train_col2 = st.columns(2, border = True)

        with optim_col2:
            optim_choice2 = st.checkbox("Tuned parameters from gridsearch", key="m23")

        with train_col2:
            train_choice2 = st.checkbox('New training', key="m24")

        if optim_choice2: 
            grid_metric_choice2 = st.selectbox('Select the gridsearch scoring metric',
                                            grid_metric_list, index=None, key="m25")

            modeling_batch = "simplegrid"

            if grid_metric_choice2:
                # Chargement du modèle correspondant
                model_dir2 = "./saved_models/global/" + dataset2 + "/" + modeling_batch + "/"

                check_model_exist2 = Path(model_dir2 + grid_metric_choice2 + clf_choice2 + "search.pkl").exists()

                if check_model_exist2:
                    st.write("Model loading from folder :", model_dir2 + grid_metric_choice2 + clf_choice2 + "search.pkl")

                    # charge le gridsearch
                    #with open(model_dir  + grid_metric_choice  + clf_choice  +"search.pkl", 'rb') as s:
                    #    search = pickle.load(s)

                    # charge le modèle
                    with open(cwd+ model_dir2  + grid_metric_choice2 + "_" + clf_choice2 +"_tuned.pkl", 'rb') as m:
                        model2 = pickle.load(m)

                    st.write("Model Parameters : \n ", model2)
                    model2_loaded = True

                else:
                    st.write("No model at :", model_dir2 + grid_metric_choice2 + clf_choice2 + "search.pkl")
                    parameters_choice2 = st.selectbox('Select your parameters ', 'Default parameters', index = None)
                    model2 = classifier_list[clf_choice2]
                    if parameters_choice2:
                        st.write("The selected model will be trained with the following parameters : \n ", model2)
                        model2.fit(X_train2, y_train2)
                        model2_loaded = True

        elif train_choice2:
            parameters_choice2 = st.selectbox('Select your parameters ', 'Default parameters', index = None, key="m26")
            model2 = classifier_list[clf_choice2]
            if parameters_choice2:
                modeling_batch = "livetrain"
                model_train_dir2 = "./saved_models/global/" + dataset2 + "/" + modeling_batch + "/"
                check_model_train_exist2 = Path(model_train_dir2 + clf_choice2 + ".pkl").exists()
                if check_model_train_exist2:
                    st.write("Model already exists. Loading it from folder :", model_train_dir2 + clf_choice2 + ".pkl")
                    with open(cwd+ model_train_dir2 +clf_choice2 +".pkl", 'rb') as m:
                        model2 = pickle.load(m)
                    model2_loaded = True
                else:
                    st.write("The selected model will be trained with the following parameters : \n ", model2) 
                    model2.fit(X_train2, y_train2)
                    model2_loaded = True


##############################
# Résultats Deuxième modèle
##############################

        if model2_loaded:
            # prédictions

            y_pred2 = model2.predict(X_test2)


            st.write("Select the results to print")

            inter_col2, cm_col2, report_col2 = st.columns(3, border = True)

            with inter_col2 :
                inter_choice2 = st.checkbox("Interpretability plot", key="m27")
            with cm_col2 :
                cm_choice2 = st.checkbox("Confusion matrix", key="m28")
            with report_col2 :
                report_choice2 = st.checkbox("Metrics", key="m214")

            if inter_choice2:
                n_feat2 = st.slider('Number of features',
                                min_value = 1,
                                max_value = len(X_train2.columns),
                                value = 10, key="m217")
            else:
                n_feat2=10

            if inter_choice2 or cm_choice2 or report_choice2:

                report2, cm2, plot_res2 = mf.st_plot_model_results(
                    model_name = clf_choice2, model= model2,
                    X_train = X_train2, X_test = X_test2,
                    y_train = y_train2, y_test =y_test2,
                    plot_inter = inter_choice2,
                    plot_cm = cm_choice2,
                    plot_report = report_choice2,
                    n_coef_graph=n_feat2,
                    )
                # st.dataframe(report)
                st.pyplot(plot_res2)

                if train_choice2 or check_model_exist2==False:
                    st.write("Do you want to save this model with pickle?")
                
                    save_model2 = st.checkbox("Save the model", key="m230")
                    # timestampnow = pd.Timestamp(datetime.datetime.now()).round(freq='min').timestamp()
                    # timestampnow = round(timestampnow,ndigits=0)
                    # pd.to_datetime(timestampnow, unit = "s")
                    if save_model2:
                        modeling_batch = "livetrain"
                        model_dir2 = "./saved_models/global/" + dataset2 + "/" + modeling_batch + "/"
                        os.makedirs(os.path.dirname(model_dir2), exist_ok=True)
                        with open(cwd + model_dir2  + clf_choice2 + ".pkl", 'wb') as m:
                                pickle.dump(model2,m)


                st.write("Select more criteria to deepen the results")
                seuil_col2, location_col2 = st.columns(2, border = True)
                with seuil_col2:
                    seuil_choice2 = st.checkbox("Classification treshold", key="m29")
                with location_col2:
                    location_choice2 = st.checkbox("Forecast by weather station", key="m210")

                if location_choice2:
                    plot_FN2, plot_FP2 = mf.st_plot_FP_FN(
                        "Location",
                        X_train2, X_test2, y_train2, y_test2,
                        y_pred2)
                    st.pyplot(plot_FN2)
                    st.pyplot(plot_FP2)

                if seuil_choice2:
                    seuil_plot2, metric_plot2 = mf.st_plot_seuil_roc_AUC(
                        model2,
                        X_train2, X_test2, y_train2, y_test2)

                    st.pyplot(seuil_plot2)
                    st.pyplot(metric_plot2)

                    threshold2 = st.slider("Treshold adjustment", 0.0, 1.0, value = 0.2, step=0.1, key="m211")
                    adjust_report2, cm_avant2, cm_apres2 = mf.adjust_classification_treshold(model2,
                                        X_train2, X_test2, y_train2, y_test2,
                                        threshold2)
                    col3, col4 = st.columns(2, border = False)
                    with col3:
                        st.dataframe(adjust_report2.iloc[:,0])
                        st.dataframe(cm_avant2)
                    with col4:
                        st.dataframe(adjust_report2.iloc[:,1])
                        st.dataframe(cm_apres2)

                st.write("Do you want to make a forecast?")
                location2, time2 = st.columns(2, border = True)
                with location2:
                    city2 = st.selectbox(label = 'Select a city',
                                options = sorted(df2.index.get_level_values(0).unique()),
                                index = None,
                                label_visibility="visible",
                                help = 'get a listed city',
                                placeholder = 'No city selected yet',
                                key="m212")

                if city2 :
                    date2 = X_test2.loc[city2,:].index.get_level_values(0)[-10]
                    with time2:
                        date2 = st.date_input(label = 'Select a Date',
                                        help = 'year-month-day',
                                        value = date2,
                                        key="m213")

                if city2 and date2:
                    datetomorrow2 = str(date2 + datetime.timedelta(days=1))
                    date2 = str(date2)
                    X_predict2 = np.array(X_test2.loc[(city2, date2),])
                    pred2 = model2.predict(X_predict2.reshape(1, -1))
                    prob2 = model2.predict_proba(X_predict2.reshape(1, -1))
                    if pred2 == 1:
                        st.write("Don't forget your umbrella, it is going to rain on the ",datetomorrow2, " à ", city2)
                        st.write("Probability : ", str(round(prob2[0,1]*100, ndigits=0)), "%")
                    if pred2 == 0:
                        st.write("No rain on the ",datetomorrow2, " à ", city2)
                        st.write("Probability : ", str(round(prob2[0,0]*100, ndigits=0)), "%")

                    y_true2 = y_test2.loc[(city2, date2),]
                    if  y_true2 == 1.0:
                        st.write("In reality, it rained at  ", city2, " on the ", datetomorrow2)
                        st.write("It rained ", str(df2.loc[(city2, datetomorrow2),"Rainfall"]), " mm")
                    if  y_true2 == 0.0:
                        st.write("In reality, it rained at  ", city2, " on the ", datetomorrow2)
                        st.write("It rained ", str(df2.loc[(city2, datetomorrow2),"Rainfall"]), " mm")
