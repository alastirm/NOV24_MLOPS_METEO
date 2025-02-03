import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin



class VoisinageNAImputer(BaseEstimator, TransformerMixin):
    '''
    Ce transformer remplit les valeurs manquantes dans un DataFrame en utilisant
    une méthode de voisinage basée sur des périodes temporelles autour de chaque NaN.
    Il remplace les NaN dans une colonne par la moyenne des valeurs voisines,
    et gère plusieurs localisations.
    '''

    def __init__(self, column, global_data_path='../data/weatherAUS.csv'):
        '''
        Initialisation avec un DataFrame global lu depuis un fichier CSV (par défaut).
        Le paramètre `column` spécifie la colonne à modifier.
        '''
        self.column = column
        self.global_data_path = global_data_path
        self.global_data = pd.read_csv(self.global_data_path)

    def fit(self, X, y=None):
        '''
        Nous n'avons pas besoin de calculer quoi que ce soit dans fit, sauf valider le DataFrame.
        '''
        self.global_data['Date'] = pd.to_datetime(self.global_data['Date'])
        return self

    def transform(self, X):
        '''
        Applique la logique de remplissage des NaN pour la colonne spécifiée
        en utilisant le global_data stocké dans l'initialisation.
        '''
        X['Date'] = pd.to_datetime(X['Date'])

        # print("Data avant transformation :")
        # print(X.head())

        X[self.column] = self.remplissage_na_voisinnage(X, column=self.column)

        # print(f"Data après transformation de {self.column}:")
        # print(X['Humidity9am'].isna().sum())

        return X

    def remplissage_na_voisinnage(self, location, column):
        '''
        Fonction de remplissage des NaN par voisinage. Elle remplace les NaN dans `column` de `location`
        en utilisant la moyenne des voisins autour de la période des NaN, ou la médiane de `global_data` si
        aucun voisin n'est trouvé.
        '''
        i = 0
        location_len = len(location)

        while i < location_len:
            if pd.isna(location[column].iloc[i]):
                periode = 0
                while (i + periode < location_len) and (pd.isna(location[column].iloc[i + periode])):
                    periode += 1

                if periode == location_len:
                    location[column] = self.global_data[column].median()
                    i += location_len
                else:
                    borne_sup = min(location_len - 1, i + periode)
                    date_debut = location["Date"].iloc[i]
                    date_fin = location["Date"].iloc[borne_sup]
                    nombre_jours = (date_fin - date_debut).days + 1

                    voisinnage = location[(location["Date"] >= date_debut - pd.Timedelta(days=nombre_jours)) &
                                          (location["Date"] <= date_fin + pd.Timedelta(days=nombre_jours)) &
                                          (~location["Date"].isna())]

                    if not voisinnage.empty:
                        moyenne_voisins = voisinnage[column].mean()
                        location.loc[location.index[i]:location.index[borne_sup], column] = moyenne_voisins
                    else:
                        location[column] = self.global_data[column].median()
                    i += periode
            else:
                i += 1

        return location[column]
