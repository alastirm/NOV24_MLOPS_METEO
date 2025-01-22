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


from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Rescaling


# métriques 
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report,accuracy_score, f1_score, confusion_matrix

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

# 
X_train.shape[0]/64

y_train.value_counts()

y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)



# modèle perceptron simple  :

model = Sequential()

model.add(Dense(units=256, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=2, activation="sigmoid"))  # deux classes 


model.summary()

metric_list = ["accuracy", 
               keras.metrics.Recall(class_id = 0, name='recall'), 
               keras.metrics.TruePositives(name='true_positives')]

metric_list2 = [keras.metrics.Recall(class_id = 1, name='recall'), 
               keras.metrics.TrueNegatives(name='true_negatives')]
metric_list2 = [keras.metrics.Recall(class_id = 1, name='recall'), 
               keras.metrics.Precision(class_id = 1, name='precision')]

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=metric_list2),

model.summary()

# modèle avec 400 epochs
n_epochs = 400

history = model.fit(X_train, y_train_enc,
         validation_split=0.1,
         epochs=n_epochs, 
         batch_size=32)



# model.save('../saved_models/model_RNN_400epochs_binaryentropy_recall_precision.keras')

history.history.items()

def plot_rnn_metrics(metric_list, n_epochs, model_history) :
    
    n_plots = len(metric_list)
    plt.figure(figsize = (40,10))
    plt.subplots(ncols = 1,   nrows = n_plots, sharex = True)
    range(len(metric_list))

    for metric, p in zip(metric_list, range(len(metric_list))):
        print(p)
        print(metric)
        # Courbe de la métrique sur l'échantillon de train
        plt.subplot(n_plots, 1,  p+1)
        plt.plot(np.arange(1 , n_epochs + 1, 1),
                 model_history.history[metric],
                 label='Training ' + metric,
                 color='blue')
        # Labels des axes
        plt.ylabel(metric)
        # Courbe de la métrique sur l'échantillon de test
        plt.plot(np.arange(1 , n_epochs + 1, 1),
                 model_history.history['val_' + metric], 
                 label='Validation ' + metric,
                 color='red')
        # Affichage de la légende
        plt.legend()
    
    plt.xlabel('Epochs')
    # Affichage de la figure
    plt.show()


plot_rnn_metrics(metric_list = ["loss", "accuracy", "recall", "true_positives"],
                  n_epochs = n_epochs, model_history = history) 

plot_rnn_metrics(metric_list = ["accuracy"],
                  n_epochs = n_epochs, model_history = history) 

plot_rnn_metrics(metric_list = [ "true_positives"],
                  n_epochs = n_epochs, model_history = history) 

plot_rnn_metrics(metric_list = ["recall", "precision"],
                  n_epochs = n_epochs, model_history = history) 

# évaluation du modèle 

model = load_model("../saved_models/model_RNN_400epochs_binaryentropy_recall_precision.keras")
model.evaluate(X_test, y_test_enc)

test_pred = model.predict(X_test)
y_test_class = y_test
y_pred_class = np.argmax(test_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))


train_pred = model.predict(X_train)
y_train_class = y_train
y_pred_train = np.argmax(train_pred, axis=1)
print(classification_report(y_train_class, y_pred_train))
print(confusion_matrix(y_train_class, y_pred_train))


sns.heatmap(confusion_matrix(y_test_class, y_pred_class), cmap='Blues', cbar=False, annot=True)

# modèle avec 400 epochs
model2 = model
model2.compile(loss="hinge",
             optimizer="adam",
             metrics=metric_list2),

n_epochs2 = 100

history2 = model2.fit(X_train, y_train_enc,
                      validation_split=0.1,
                      epochs=n_epochs2 , 
                      batch_size=32)

history2.history["recall"]

history2.history["accuracy"]

plot_rnn_metrics(metric_list = ["precision", "recall"], n_epochs = n_epochs2, model_history = history2) 


model2.save('../saved_models/model_RNN_100epochs_hingr_recall_precision.keras')

model2 = load_model('../saved_models/model_RNN_100epochs_hingr_recall_precision.keras')
model2.evaluate(X_test, y_test_enc)


# évaluation du modèle 

test_pred = model2.predict(X_test)
y_test_class = y_test
y_pred_class = np.argmax(test_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

sns.heatmap(confusion_matrix(y_test_class, y_pred_class), cmap='Blues', cbar=False, annot=True)
