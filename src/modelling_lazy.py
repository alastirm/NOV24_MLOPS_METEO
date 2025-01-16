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
import xgboost as xgb
import shap
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report

# Rééchantillonage
from imblearn.over_sampling import SMOTE

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

classifiers_list = [est for est in all_estimators() if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))]
# classifier lazy 
clf_lazy = LazyClassifier(verbose=0, classifiers= classifiers_list,
                          ignore_warnings=True, 
                          custom_metric=None)

models,predictions = clf_lazy.fit(X_train, X_test, 
                                  y_train, y_test)


# print models 
display(models)

# sort par f1 score 
models = models.sort_values("F1 Score", ascending = False)
models = models.apply(lambda x: round(x, ndigits=2))

models.to_csv("../modeling_results/lazy_predict_mixtepreprocessing.csv",  decimal = ",")
