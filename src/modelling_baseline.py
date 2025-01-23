import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# features selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_regression, mutual_info_classif, RFE, RFECV

# modèles 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import shap

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report,accuracy_score, f1_score, confusion_matrix, precision_score

# Rééchantillonage
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Autres fonctions
import functions_created

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

# Choix de 3 modèles 
dt = DecisionTreeClassifier(max_depth = 1)


models_select  = {
    'LogisticRegression': LogisticRegression(max_iter = 500, n_jobs=-1),
    'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
    'BaggingClassifier': BaggingClassifier(dt, n_estimators = 500, max_samples = 0.5, max_features = 0.5, n_jobs=-1)
}


##################
# FIT BASIQUE ####
##################

# Dictionnaire pour stocker les résultats

model = LogisticRegression()

def fit_models(models_select, X_train, y_train, X_test, y_test, unbalanced = False): 
      results = {}
      cm = {}
      for model_name, model in models_select.items():
            print(model_name)
            model.fit(X_train, y_train["RainTomorrow"])

            # Tester sur les données de test
            y_pred = model.predict(X_test)

            # métriques
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1_score = f1_score(y_test, y_pred)
            # print(classification_report(y_test["RainTomorrow"], y_pred))
            report = classification_report(y_test["RainTomorrow"], y_pred, output_dict=True)
            
            if unbalanced == True :
                 report = classification_report_imbalanced(y_test["RainTomorrow"], y_pred, output_dict=True)
            
            report = pd.DataFrame(report).transpose()
            # Stocker les résultats
            results[model_name] = report
            cm[model_name] = confusion_matrix(y_test["RainTomorrow"], y_pred,)
            
      
      return results, cm

report_base, cm_base = fit_models(models_select, X_train, y_train, X_test, y_test)

# affiche et save les résultats
for model_name, model in models_select.items():
      print(model_name)
      print(report_base[model_name])
      print(cm_base[model_name])
      report_base[model_name] = report_base[model_name].apply(lambda x: round(x,ndigits=2))
      report_base[model_name].to_csv("../modeling_results/base_results" + model_name + ".csv", decimal =",")


####################################
# FIT AVEC FEATURES SELECTION ####
####################################
k_select = len(X_train.columns)//2
# Sélection des features sur la base de l'information mutuelle

sel_kbi  = SelectKBest(score_func = mutual_info_classif, k= k_select)
sel_kbi.fit(X_train, y_train["RainTomorrow"])
mask_kbi = sel_kbi.get_support()
plt.matshow(mask_kbi.reshape(1,-1), cmap = 'gray_r');
plt.xlabel('Axe des features')
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

X_train_kbi = sel_kbi.transform(X_train)
X_test_kbi = sel_kbi.transform(X_test)

report_kbi, cm_kbi = fit_models(models_select, X_train_kbi, y_train, X_test_kbi, y_test)

# affiche et save les résultats
for model_name, model in models_select.items():
      print(model_name)
      print(report_base[model_name])
      print(cm_base[model_name])
      report_base[model_name] = report_base[model_name].apply(lambda x: round(x,ndigits=2))
      report_base[model_name].to_csv("../modeling_results/kbi19_results" + model_name + ".csv", decimal =",")



##################
# GRID SEARCH ####
##################

# Hyperparamètres à tester pour chaque modèle
param_grids = {
    'LogisticRegression': {
                           'C': [0.01, 0.1, 1, 10], 
                           'solver': ['liblinear', 'lbfgs']},
    'RandomForestClassifier': {'n_estimators': [50, 100, 200], 
                               'max_depth': [10, 20, 30]},
     'BaggingClassifier': {'n_estimators': [500, 1000, 2000], 
                           'max_samples' : [0.05, 0.1, 0.2, 0.5]}
}

param_grids2 = {
    'LogisticRegression': {
        'max_iter' : [100,500],
        'C':[0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cholesky','newton-cg','lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty':['none', 'elasticnet', 'l1', 'l2'],
        'fit_intercept': [True, False]},
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'max_depth': [10, 30, 50],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]},
    'BaggingClassifier': {
        'n_estimators': [50, 500, 1000, 2000], 
        'max_samples' : [0.05, 0.1, 0.2, 0.5, 1],
        'max_features': [0.5, 0.8, 0.95, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'oob_score': [True, False]}
}


# Dictionnaire pour stocker les résultats (njobs -1)
results_search = {}

def compare_search_methods(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    # search_methods = {
    #     'GridSearchCV': GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1),
    #     'RandomizedSearchCV': RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1),
    #     'BayesSearchCV': BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    # }
    
    search_methods = {
        'GridSearchCV': GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='precision'),
    }

    results_search[model_name] = {}
    i =1
    print(model_name)
    for search_name, search in search_methods.items():
        print(search_name)
        # Effectuer la recherche d'hyperparamètres
        search.fit(X_train, y_train["RainTomorrow"])
        
        # Meilleur score et hyperparamètres trouvés
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Tester sur les données de test
        y_pred = search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_score = f1_score(y_test["RainTomorrow"], y_pred)
        test_precision = precision_score(y_test["RainTomorrow"], y_pred)

        # Stocker les résultats
        results_search[model_name][search_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1_score,
            'test_precision' : test_precision
        }


# Exécuter la comparaison pour chaque modèle
for model_name, model in models_select.items():
    compare_search_methods(model_name, model, param_grids[model_name],  X_train, y_train, X_test, y_test)

# Afficher les résultats
for model_name, model_results in results_search.items():
    print(f"Model: {model_name}")
    for search_name, search_results in model_results.items():
        print(f"  {search_name}:")
        print(f"    Best Params: {search_results['best_params']}")
        print(f"    Best CV Score: {search_results['best_cv_score']:2f}")
        print(f"    Test Accuracy: {search_results['test_accuracy']:.2f}")
    print("\n")



models_best = {
    'LogisticRegression': LogisticRegression(max_iter=100, C=0.01, solver='lbfgs', n_jobs=-1),
    'RandomForestClassifier': RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=-1),
    'BaggingClassifier': BaggingClassifier(n_estimators=500, max_samples=0.05, n_jobs=-1)
}

report_best, cm_best = fit_models(models_best, X_train, y_train, X_test, y_test)

# affiche et save les résultats
for model_name, model in models_best.items():
      print(model_name)
      print(report_best[model_name])
      print(cm_best[model_name])
      report_best[model_name] = report_best[model_name].apply(lambda x: round(x,ndigits=2))
      report_best[model_name].to_csv("../modeling_results/best_results" + model_name + ".csv", decimal =",")

#######################################################
# Gestion du déséquilibre et resampling  ##
#######################################################

# On utilise le rééchantillonnage pour traiter le déséquilibre de la variable cible -> SMOTE

print('Classes échantillon initial :', dict(pd.Series(y_train["RainTomorrow"]).value_counts()))

smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm["RainTomorrow"]).value_counts()))


# Gridsearch avec X_sm et y_sm
for model_name, model in models_select.items():
    compare_search_methods(model_name, model, param_grids2[model_name],  X_sm, y_sm, X_test, y_test)

# Afficher les résultats
for model_name, model_results in results_search.items():
    print(f"Model: {model_name}")
    for search_name, search_results in model_results.items():
        print(f"  {search_name}:")
        print(f"    Best Params: {search_results['best_params']}")
        print(f"    Best CV Score: {search_results['best_cv_score']:2f}")
        print(f"    Test Accuracy: {search_results['test_accuracy']:.2f}")
    print("\n")


models_best_sm =  {
    'LogisticRegression': LogisticRegression(max_iter=100, C=10, solver='liblinear', n_jobs=-1),
    'RandomForestClassifier': RandomForestClassifier(max_depth=30, n_estimators=100, n_jobs=-1),
    'BaggingClassifier': BaggingClassifier(n_estimators=2000, max_samples=0.05, n_jobs=-1)
}

report_sm, cm_sm = fit_models(models_best_sm, X_sm, y_sm, X_test, y_test)

# affiche et save les résultats
for model_name, model in models_select.items():
      print(model_name)
      print(report_sm[model_name])
      print(cm_sm[model_name])
      report_sm[model_name] = report_sm[model_name].apply(lambda x: round(x,ndigits=2))
      report_sm[model_name].to_csv("../modeling_results/smote_results" + model_name + ".csv", decimal =",")


# idem avec UnderSampler

unsam = RandomUnderSampler()
X_sm, y_sm = unsam.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm["RainTomorrow"]).value_counts()))

# Gridsearch avec X_sm et y_sm
for model_name, model in models_select.items():
    compare_search_methods(model_name, model, param_grids[model_name],  X_sm, y_sm, X_test, y_test)


########################
# MODELES PAR STATION
########################

# Teste un modèle sur une location

location = "Sydney"
report_loc = {}
cm_loc = {}

report_loc[location], cm_loc[location] = fit_models(models_best, 
                                                   X_train.loc[location], 
                                                   y_train.loc[location], 
                                                   X_test.loc[location], 
                                                   y_test.loc[location])

for model_name, model in models_select.items():
      print(model_name)
      print(report_loc[location][model_name])
      print(cm_loc[location][model_name])
      report_loc[location][model_name] = report_loc[location][model_name].apply(lambda x: round(x,ndigits=2))
      report_loc[location][model_name].to_csv(
           "../modeling_results/location/location_results_" + model_name + "_" + location + ".csv", decimal =",")


##################
# VRAC A TRIER
##################


# Pour le fun, XGboost et valeurs de shapley

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test["RainTomorrow"], y_pred_xgb))

explainer = shap.Explainer(xgb_clf)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type='bar')

xgb_clf_sm = xgb.XGBClassifier()
xgb_clf_sm.fit(X_sm, y_sm)

y_pred_xgb_sm = xgb_clf_sm.predict(X_test)
print(classification_report(y_test["RainTomorrow"], y_pred_xgb_sm))

explainer_sm = shap.Explainer(xgb_clf_sm)
shap_values_sm = explainer_sm.shap_values(X_train)

shap.summary_plot(shap_values_sm, X_sm, plot_type='bar')
