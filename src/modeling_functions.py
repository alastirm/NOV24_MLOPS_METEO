import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# lazy 
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

# search hyperparameters
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

# modèles 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

import xgboost as xgb
import shap

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report,accuracy_score, f1_score, fbeta_score
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, precision_recall_curve

# Rééchantillonage
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from imblearn.under_sampling import RandomUnderSampler

# scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# sauvegarde model
import pickle
import itertools

# fonction de séparation test / train selon deux méthodes 

def separation_train_test(df, sep_method = "classique", col_target = "RainTomorrow"):
    # drop les Nas restants 
    df = df.dropna()
    target = df[col_target]
    feats = df.drop(columns = col_target)

    if sep_method == "classique":
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=1234, stratify=target)
    if sep_method == "temporelle":
        df["Date"] = df.index.get_level_values(1).values
        years = pd.to_datetime(df.Date).dt.year
        years.unique()
        years.value_counts(ascending=True)
        Part_data_year = years.value_counts().sort_index().cumsum()/len(years)
        print("Part des données par années ", Part_data_year)
        year_split = Part_data_year.loc[Part_data_year > 0.8].index[0]
        id_train = years[years < year_split].index
        id_test = years[years >= year_split].index
        print("Séparation des données avant et après ", year_split)
        X_train = feats.loc[id_train]
        X_test =  feats.loc[id_test]
        y_train = target.loc[id_train]
        y_test =  target.loc[id_test]

    if 'Date' in X_train.columns:
        X_train = X_train.drop(columns = "Date")
        X_test = X_test.drop(columns = "Date")
    if 'Location' in X_train.columns:
        X_train = X_train.drop(columns = 'Location')
        X_test = X_test.drop(columns = 'Location')

    return X_train, X_test, y_train, y_test


# fonction pour scaler, avec choix du scaler

def scaling(X_train, X_test, scaler = MinMaxScaler()):

    scaler = scaler
    # On fit sur Xtrain complet
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)
    
    return X_train_scaled, X_test_scaled

# fonction pour fit un lazy predict

removed_classifiers = [
"ClassifierChain",
"ComplementNB",
"GradientBoostingClassifier",
"GaussianProcessClassifier",
"HistGradientBoostingClassifier",
"MLPClassifier",
"LogisticRegressionCV",
"MultiOutputClassifier",
"MultinomialNB",
"OneVsOneClassifier",
"OneVsRestClassifier",
"OutputCodeClassifier",
"RadiusNeighborsClassifier",
"VotingClassifier",
'SVC','LabelPropagation','LabelSpreading','NuSV']

def lazy_results(X_train, X_test, y_train, y_test, removed_classifiers=removed_classifiers):
    classifiers_list = [est for est in all_estimators() if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))]
    # création classifier lazy 
    clf_lazy = LazyClassifier(
        verbose=0, classifiers= classifiers_list,
        ignore_warnings=True, 
        custom_metric=None)

    # fits
    models, predictions = clf_lazy.fit(X_train, X_test, y_train, y_test)
    # sort par f1 score 
    models= models.sort_values("F1 Score", ascending = False)
    models = models.apply(lambda x: round(x, ndigits=2))
    # print models 
    display(models)
    return models, predictions


# fonction pour fit une liste de modèle et sauvegarder les modèles 

def fit_models(models_select, 
               X_train,X_test, y_train, y_test, 
               unbalanced = False, 
               save_model = True, save_models_dir = "",
               save_results = True, save_results_dir = ""): 
      

    # création de  dictionnaires pour stocker les résultats 

    results = {} 
    cm = {}
    fitted_models = {}

    for model_name, model in models_select.items():
        print("Fitting ", model_name)
        model.fit(X_train, y_train)

        # Tester sur les données de test
        y_pred = model.predict(X_test)

        # métriques
        # test_accuracy = accuracy_score(y_test, y_pred)
        # test_f1_score = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
            
        if unbalanced:
            report = classification_report_imbalanced(y_test, y_pred, output_dict=True)
            
        report = pd.DataFrame(report).transpose()
        # Stocker les résultats
        results[model_name] = report
        cm[model_name] = confusion_matrix(y_test, y_pred)
        cm[model_name] = pd.DataFrame(cm[model_name], index = y_train.unique(), 
                                          columns=y_train.unique())    
            
        

        fitted_models[model_name] = model
        
        # sauvegarde les modèles en pickle (si true)
            
        if save_model:     
            with open("../saved_models/" + save_models_dir + model_name + '.pkl', 'wb') as f:
                pickle.dump(model, f)

        # sauvegarde les résultas en csv (si true)

        if save_results:   
            results[model_name] = results[model_name].apply(lambda x: round(x, ndigits=2))
            results[model_name].to_csv("../modeling_results/" + 
                                       save_results_dir + model_name + ".csv", 
                                       decimal =",")
   
    # print les résultats

    for model_name, model in models_select.items():
        print("\n")
        print("Nom du modèle :", model_name)
        print("Rapport de classification :")
        print(results[model_name])
        print("Matrice de confusion :")
        print(cm[model_name])

    return results, cm, fitted_models

def print_save_models(models_select, report,  cm,  save_name, dir_name=''):
    # affiche et save les résultats
    for model_name, model in models_select.items():
        print("\n")
        print("Nom du modèle :",model_name)
        print("Rapport de classification :")
        print(report[model_name])
        print("Matrice de confusion :")
        print(cm[model_name])
        report[model_name] = report[model_name].apply(lambda x: round(x,ndigits=2))
        report[model_name].to_csv("../modeling_results/" + dir_name + "/results_" + save_name + "_" + model_name + ".csv", decimal =",")


# Fonction pour rééchantilloner avec choix du resampler

def resample(X_train, y_train, resampler = SMOTE()):
    
    print('Classes échantillon initial :', dict(pd.Series(y_train).value_counts()))
    resampler = resampler
    X_sm, y_sm = resampler.fit_resample(X_train, y_train)
    print('Classes échantillon rééchantillonné :', dict(pd.Series(y_sm.value_counts())))
    return X_sm, y_sm


# Fonction d'optimisation deshyperparamètres

# Dictionnaire pour stocker les résultats (njobs -1)

def optimize_parameters(model_name, model, 
                        param_grid, 
                        X_train,  X_test, y_train, y_test,
                        scoring='average_precision',
                        search_method = 'GridSearchCV'):
    # search_methods = {
    #     'GridSearchCV': GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1),
    #     'RandomizedSearchCV': RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1),
    #     'BayesSearchCV': BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=5, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    # }
    
    if search_method == 'GridSearchCV':

        search_methods = {
            'GridSearchCV': GridSearchCV(
                estimator=model, 
                param_grid=param_grid, cv=3, 
                scoring=scoring, refit = "accuracy",
                verbose =4, n_jobs = -1,
                return_train_score=True),
        }

    if search_method == 'BayesSearchCV':

        search_methods = {
            'BayesSearchCV': BayesSearchCV(
                estimator=model, 
                search_spaces=param_grid, cv=3, n_iter=200, 
                scoring=scoring, refit = "accuracy",
                verbose =4, n_jobs = -1,
                return_train_score=True,
                random_state=1234),
        }

    results_search = {}
    results_search[model_name] = {}
    i = 1
    print("Optimizing" , model_name)
    print("\n")
    for search_name, search in search_methods.items():
        print(search_name)
        # Effectuer la recherche d'hyperparamètres
        search.fit(X_train, y_train)
        
        # Meilleur score et hyperparamètres trouvés
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Tester sur les données de test
        y_pred = search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_score = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_search[model_name][search_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1_score,
            'test_precision' : test_precision,
            'test_recall' : test_recall
        }

    return results_search, search



# fonction pour modéliser l'ensemble de la chaîne sur une location

def modeling_location(select_location, 
                      df_location, 
                      models_select, 
                      param_grids, scoring = "accuracy",
                      resampler = RandomUnderSampler(), 
                      save_name="") :

    X_train, X_test, y_train, y_test = \
        separation_train_test(df_location, 
                            sep_method = "temporelle", 
                            col_target = "RainTomorrow")


    X_train_scaled, X_test_scaled = scaling(X_train, X_test, scaler = MinMaxScaler())

    # lazy 

    models_lazy, predictions_lazy = lazy_results(X_train_scaled, X_test_scaled, y_train, y_test)

    # fit modèles simples

    report_location, cm_location, models_location = \
        fit_models(models_select,
        X_train_scaled, X_test_scaled, y_train, y_test,
        save_model=True,
        save_models_dir="../saved_models/location/"+ save_name + select_location + "_base_",
        save_results = False)


    # # undersampling et gridsearch sur données preprocessed V2
    # X_train_rs, y_train_rs = resample(X_train_scaled,  y_train, resampler = resampler)

    best_models = {}

    # Exécuter le search pour chaque modèle
    for model_name, model in models_select.items():
        print("Optimizing", model_name)
        results_search, search = optimize_parameters(
            model_name, model, 
            param_grids[model_name],  
            X_train_scaled, X_test_scaled, y_train,  y_test,
            scoring = scoring)
        
        best_models[model_name] = search.best_estimator_

    print(best_models)
    # Fit et sauvegarde les meilleurs modèle du GridSearch

    report_location, cm_location, models_location = \
        fit_models(best_models, 
        X_train_scaled, X_test_scaled,y_train, y_test,
        save_model=True,
        save_models_dir="../saved_models/location/"+ save_name + select_location + "_best_",
        save_results=False)
    

    for model_name, model in models_select.items():
    

        model_dir = "../saved_models/location/" + save_name + select_location + "_best_" + model_name
        with open(model_dir + '.pkl', 'rb') as f:
            model = pickle.load(f)

        # Graph des coeff de la LogisticRegression
        if type(model).__name__ == 'LogisticRegression': 

            pd.Series(model.coef_[0], 
                    X_train.columns).sort_values(ascending=False)\
                        .plot(kind='barh', figsize=(15,20));
            plt.title("Coefficients " + select_location + " " + model_name)

            graph_dir = "../modeling_results/location/"+ save_name + select_location + "_best_" + model_name
            plt.savefig(graph_dir + ".png")

        # Graph des Features importances pour le RandomForest
        if (type(model).__name__ == 'RandomForestClassifier') or (type(model).__name__ == "BalancedRandomForestClassifier"):  

            # features importances 
            feat_importances = pd.Series(
                model.feature_importances_, index=X_train.columns)
            fig = plt.figure(figsize = (12,10))
            feat_importances.nlargest(20).plot(kind='barh');
            plt.title("Features importance " + select_location + " " + model_name)

            graph_dir = "../modeling_results/location/"+ save_name + select_location + "_best_" + model_name
            fig.savefig(graph_dir + ".png")

    return results_search, search


def plot_model_results(model_name, model_dir, graph_dir, 
                       graph_title,
                       X_train, X_test, y_train, y_test):
    

    with open(model_dir + '.pkl', 'rb') as f:
        model = pickle.load(f)

    list_model = {}
    list_model[model_name] = model
    report, cm, models = \
        fit_models(list_model, 
        X_train, X_test,y_train, y_test,
        save_model=False,
        save_results=False)
    

    report[model_name] = report[model_name].apply(lambda x: round(x, ndigits=2))
    report[model_name].loc["accuracy"] =[report[model_name].loc["accuracy","precision"],"","",""]
    plt.figure(figsize=(16, 12))
    # Graph des coeff de la LogisticRegression
    if type(model).__name__ == 'LogisticRegression': 
            serie_coef = pd.Series(model.coef_[0], 
                    X_train.columns).sort_values(ascending=False)
            index_toplot = list(np.arange(0, 10, 1))
            for x in list(np.arange(-10, -1, 1)):
                    index_toplot.append(x) 
                    serie_coef.iloc[index_toplot].plot(kind='barh', figsize=(15,20));
            plt.title("Coefficients " + graph_title)
            


    if (type(model).__name__ == 'RandomForestClassifier') or (type(model).__name__ == "BalancedRandomForestClassifier"):  
            # features importances 
            feat_importances = pd.Series(
            model.feature_importances_, index=X_train.columns)
            feat_importances.nlargest(20).plot(kind='barh');
            plt.title("Features importance " + graph_title)

    plt.subplots_adjust(left=0.2, bottom=0.4)
    table_cm = plt.table(cellText=cm[model_name].values,
            rowLabels=cm[model_name].index,
            colLabels=cm[model_name].columns,
            cellLoc = 'center', rowLoc = 'center',
            transform=plt.gcf().transFigure,
            bbox = ([0.1, 0.25, 0.3, 0.1]))
    table_cm.auto_set_font_size(False)
    table_cm.set_fontsize(10)
            
    table_report = plt.table(cellText=report[model_name].values,
            rowLabels=report[model_name].index,
            colLabels=report[model_name].columns,
            cellLoc = 'center', rowLoc = 'center',
            transform=plt.gcf().transFigure,
            bbox = ([0.6, 0.25, 0.3, 0.1]))
    table_report.auto_set_font_size(False)
    table_report.set_fontsize(10)
    
    plt.savefig(graph_dir + ".png", bbox_inches="tight")



def compare_model_results(modeling_batch, model_qual, model_list, 
                          location_list,metrics_list,class_label):
    
    index_results = list(itertools.product(location_list, model_list))
    index_results = pd.MultiIndex.from_tuples(index_results , 
                                          names=["Location", "model_name"])
    table_results = pd.DataFrame(index = index_results, columns=[metrics_list] )
    table_results["accuracy"]=""

    for model_name in model_list:

        for select_location in location_list:
            
            df_location =  pd.read_csv("../src/data_location_V2/df_" + select_location + ".csv", index_col=["id_Location","id_Date"])
            
            X_train, X_test, y_train, y_test = \
                separation_train_test(df_location, 
                                    sep_method = "temporelle", 
                                    col_target = "RainTomorrow")
            X_train_scaled, X_test_scaled = scaling(X_train, X_test, scaler = MinMaxScaler())
            
            model_dir =  "../saved_models/location/" + modeling_batch + \
            select_location + "_" + model_qual + "_" + model_name
            
            with open(model_dir + '.pkl', 'rb') as f:
                model = pickle.load(f)

            list_model = {}
            list_model[model_name] = model
            report, cm, models = fit_models(list_model, 
                X_train, X_test, y_train, y_test,
                save_model=False,
                save_results=False)

            table_results.loc[(select_location,model_name),"accuracy"] = \
                report[model_name].loc["accuracy","precision"]
            for m in metrics_list:
                table_results.loc[(select_location,model_name),m] = \
                report[model_name].loc[class_label,m]


    return table_results
