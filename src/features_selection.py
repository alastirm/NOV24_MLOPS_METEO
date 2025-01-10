import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
import umap

from sklearn.ensemble import RandomForestClassifier

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

#####################
# Variance treshold #
#####################

# Sélection des features avec une variance suffisante

sel_vt = VarianceThreshold(threshold=0.01)
sel_vt.fit(X_train)

# plot des features supprimées
mask_vt = sel_vt.get_support()
plt.matshow(mask_vt.reshape(1, -1), cmap='gray_r')
plt.xlabel('Axe des features')
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)

X_train_VT = sel_vt.transform(X_train)
X_test_VT = sel_vt.transform(X_test)

################
# Select Kbest #
################
# On garde environ la moitié des variables (valeur modifiable)
k_select = len(X_train.columns)//2

# Sélection des features sur la base d'un test de ficher


sel_kbf = SelectKBest(score_func=f_regression, k=k_select)
sel_kbf.fit(X_train, y_train)
mask_kbf = sel_kbf.get_support()
# X_train.columns[mask_kbf]

plt.matshow(mask_kbf.reshape(1, -1), cmap='gray_r')
plt.xlabel('Axe des features')
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()
X_train_kbf = sel_kbf.transform(X_train,)
X_test_kbf = sel_kbf.transform(X_test)

# Sélection des features sur la base de l'information mutuelle

sel_kbi  = SelectKBest(score_func = mutual_info_regression, k=k_select)
sel_kbi.fit(X_train, y_train)
mask_kbi = sel_kbi.get_support()
plt.matshow(mask_kbi.reshape(1,-1), cmap = 'gray_r');
plt.xlabel('Axe des features')
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

#################################
# Recursive Feature Elimination #
#################################

# Attention à appliquer sur des données scaled !

lr = LogisticRegression()
rfe = RFE(estimator=lr, step=1, n_features_to_select = k_select)
rfe.fit(X_train, y_train)

mask_rfe = rfe.get_support()
plt.matshow(mask_rfe.reshape(1,-1), cmap = 'gray_r')
plt.xlabel('Axe des features')
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

ranking_rfe = rfe.ranking_
plt.matshow(ranking_rfe.reshape(1,-1), cmap = 'gray_r')
plt.xlabel('Axe des features');
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

# RFE avec une cross validation (pas besoin de k_select)

crossval = KFold(n_splits = 5, random_state = 2, shuffle = True)
rfecv = RFECV(estimator=lr, cv = crossval, step=1)
rfecv.fit(X_train, y_train)

mask_rfecv = rfecv.get_support()
plt.matshow(mask_rfecv.reshape(1,-1), cmap = 'gray_r')
plt.xlabel('Axe des features');
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

print("Nombre de features retenues :", rfecv.n_features_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

for i in range(5):
    ax1.plot(rfecv.cv_results_[f'split{i}_test_score'])
    ax1.set_xlabel('Nombre de features')
    ax1.set_ylabel('Score')
    ax1.set_title('Score par fold de test pour chaque itération')
    
    ax2.plot(rfecv.cv_results_['mean_test_score'])
    ax2.set_xlabel('Nombre de features')
    ax2.set_ylabel('Score')
    ax2.set_title('Score moyen en cross validation')

plt.savefig("../plots/features_selection/RFE_LR_CV_score.png")
plt.show();

#########
# Lasso #
#########

lasso = Lasso(alpha = 1)
model = SelectFromModel(estimator = lasso, threshold = 1e-10)
model.fit(X_train, y_train)

mask_lasso = model.get_support()
plt.matshow(mask_lasso.reshape(1,-1), cmap = 'gray_r')
plt.xlabel('Axe des features');
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()

# recherche du meilleur paramètre de régularisation (alpha entre 0 et 1)

alpha_grid = {'alpha': np.arange(0.01,1,0.02)}
grid_lasso = GridSearchCV(estimator = lasso, 
                    param_grid = alpha_grid, 
                    cv=crossval, 
                    scoring = 'neg_mean_squared_error')
grid_lasso.fit(X_train, y_train)
print(grid_lasso.best_params_)

sel_lasso_best = SelectFromModel(estimator = grid_lasso.best_estimator_, threshold = 1e-10, prefit = True)
mask_lasso_best = sel_lasso_best.get_support()
plt.matshow(mask_lasso_best.reshape(1,-1), cmap = 'gray_r')
plt.xlabel('Axe des features');
plt.xticks(rotation=90,
           ticks=range(len(X_train.columns)),
           labels=X_train.columns)
plt.show()


#######
# ACP #
#######

# projection sur 2 composantes
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train_2D[:, 0], X_train_2D[:, 1], 
           c = y_train["RainTomorrow"], 
           cmap=plt.cm.Spectral)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title("Données projetées sur les 2 axes de PCA")
plt.show()

print("La part de variance expliquée est", 
      round(pca.explained_variance_ratio_.sum(),2))


# évolution de la variance expliquée en fonction du nombre de composantes
pca = PCA()
pca.fit(X_train)

plt.figure()
plt.xlim(0,100)
plt.plot(pca.explained_variance_ratio_);

plt.figure()
plt.xlim(0, 100)
plt.xlabel('Nombre de composantes')
plt.ylabel('Part de variance expliquée')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.plot(pca.explained_variance_ratio_.cumsum());
plt.savefig("../plots/features_selection/pca_explained_variance_ratio.png")
plt.show()

explained_variance = pd.DataFrame(pca.explained_variance_ratio_.cumsum())

n_components_90 = explained_variance[explained_variance[0] > 0.90].index[0]

print("La variance expliquée est supérieure à 90 % pour", n_components_90 + 1, 
      "composantes")

pca = PCA(n_components=0.9)
pca.fit(X_train)

print("Nombre de composantes retenues :", pca.n_components_)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# comparaison avec et sans ACP pour un modèle basique

clf = RandomForestClassifier(n_jobs = -1)
# L'argument n_jobs ne vaut pas -1 par défaut. Cette valeur permet de forcer le processeur à utiliser toute sa puissance de calcul parallèle.
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf_pca = RandomForestClassifier(n_jobs = -1)
clf_pca.fit(X_train_pca, y_train)
clf_pca.score(X_test_pca, y_test)


# Si le score est proches, la PCA est efficace pour réduire la dimension en conservant l'info

# Part de la variance expliquée par Axe

variance_expliquee = pca.explained_variance_ratio_
Axes = ["Axe " + str(i+1) for i in range(len(variance_expliquee))]

plt.bar(range(len(variance_expliquee)), variance_expliquee)
plt.xticks(range(len(variance_expliquee)), Axes, rotation=90)
plt.xlabel('Composante Principale')
plt.ylabel('Part de Variance Expliquée')
plt.title('Part de Variance Expliquée par Composante Principale')
plt.show()


# Corrélation avec les features
charges_factorielles = pca.components_
df_charges_factorielles = pd.DataFrame(charges_factorielles, 
                                       columns=X_train.columns, 
                                       index=Axes)

display(df_charges_factorielles)

# fonction pour tracer le cercle de corrélation sur deux axes principaux

functions_created.draw_correlation_circle(df_charges_factorielles, pca, 
                                          Axes_names=['Axe 2', 'Axe 3'])

functions_created.draw_correlation_circle(df_charges_factorielles, pca, 
                                          Axes_names=['Axe 3', 'Axe 4'])

functions_created.draw_correlation_circle(df_charges_factorielles, pca, 
                                          Axes_names=['Axe 4', 'Axe 5'])

# heatmap des corrélations entre composantes et features
plt.figure(figsize=(20,12))
sns.heatmap(df_charges_factorielles.iloc[0:10], annot=True, cmap='viridis');


pca_mat = pd.DataFrame({'Axe 1': X_train_pca[:, 0], 
                        'Axe 2': X_train_pca[:, 1], 
                        'target': y_train["RainTomorrow"]})

plt.figure(figsize=(10, 10))
sns.scatterplot(x='Axe 1', y='Axe 2', 
                hue="target", data=pca_mat);


#######
# LDA #
#######

# Intéressant quand on prédit plus de 2 classes...

lda = LDA()
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
X_train_lda.shape


clf_lda = RandomForestClassifier(n_jobs = -1)
clf_lda.fit(X_train_lda, y_train)
clf_lda.score(X_test_lda, y_test)


# #####################
# # Manifold learning #
# #####################

# Test sur une station car c'est très long

X_train = X_train.loc["Canberra"]
y_train = y_train.loc["Canberra"]

# LLE

lle = LocallyLinearEmbedding(n_neighbors=50, 
                             n_components=2, 
                             method='modified', 
                             random_state = 42)
X_train_lle = lle.fit_transform(X_train)

fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(X_train_lle[:, 0], 
           X_train_lle[:, 1],  
           c = y_train["RainTomorrow"], 
           cmap=plt.cm.Spectral, alpha = .7, s = 4)

ax.set_xlabel('LL 1')
ax.set_ylabel('LL 2')
ax.set_title("Manifold 2D identifié par la LLE")
plt.show()

# ISomap 

isomap = Isomap(n_neighbors=50, n_components=2)
X_train_ISO = isomap.fit_transform(X_train)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train_ISO[:, 0], 
           X_train_ISO[:, 1],  
           c = y_train["RainTomorrow"], 
           cmap=plt.cm.Spectral, alpha = .7, s = 4)

ax.set_title("Données projetées sur les 2 composantes de Isomap")
plt.show();

# TSNE

tsne = TSNE(n_components=2, method = 'barnes_hut')
X_train_TSNE = tsne.fit_transform(X_train)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train_TSNE[:, 0], X_train_TSNE[:, 1],  
           c = y_train["RainTomorrow"], 
           cmap=plt.cm.Spectral, alpha = .7, s = 4)

ax.set_title("Données projetées sur les 2 axes de Tsne")
plt.show()

# UMAP

umap_model = umap.UMAP(n_neighbors = 15, min_dist = 0.1, n_components=2)
embedding = umap_model.fit_transform(X_train)

plt.figure(figsize = (10,8))
plt.scatter(embedding[:, 0], embedding[:, 1], 
            y_train["RainTomorrow"])
plt.title("Après UMAP")
plt.show()