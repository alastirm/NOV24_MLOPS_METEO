import geopy.distance
import pandas as pd
import numpy as np

class rainfall_raintoday_transformer():

    def __init__(self, city, variable="Rainfall", distance_max=50):
        self.city = city
        self.variable = variable
        self.distance_max = distance_max
        self.col_rainfall = ["Rainfall"]
        self.col_raintoday = ["RainToday"]
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_na_rainfall = X.dropna(subset=self.col_rainfall)
        missing_percentage_rainfall = (1 - len(X_na_rainfall) / len(X)) * 100
        if missing_percentage_rainfall > 30:
            X = X.drop(columns=self.col_rainfall)
            return f'{self.col_rainfall} supprimé'
        X_na_raintoday = X.dropna(subset=self.col_raintoday)
        missing_percentage_raintoday = (1 - len(X_na_raintoday) / len(X)) * 100

        if missing_percentage_raintoday > 30:
            X = X.drop(columns=self.col_raintoday)
            return f'{self.col_raintoday} supprimé'



        # Discrétisation de RainToday (si la colonne existe encore)
        if 'RainToday' in X.columns:
            X['RainToday'] = X['RainToday'].replace(['Yes', 'No'], [1, 0])
            X['RainToday'] = X['RainToday'].fillna(0).astype(int)

        # Remplir les NaN pour Rainfall avec les valeurs de la station à proximité (moins de 50 km)
        if 'Rainfall' in X.columns:
            Rainfall_near = self.complete_na_neareststation(X, variable=self.variable, distance_max=self.distance_max)

            # Ajouter les valeurs récupérées au DataFrame
            X = pd.merge(X, Rainfall_near[[self.variable + "_near"]], left_index=True, right_index=True, how="left")

            # Remplir les NaN de Rainfall avec les valeurs de Rainfall_near
            X.loc[(X["Rainfall"].isna()) & ~(X["Rainfall_near"].isna()), "Rainfall"] = \
                X.loc[(X["Rainfall"].isna()) & ~(X["Rainfall_near"].isna()), "Rainfall_near"]

            # Supprimer la colonne temporaire Rainfall_near
            X = X.drop(columns="Rainfall_near")

        # Mettre à jour RainToday en fonction des nouvelles valeurs de Rainfall
        if 'RainToday' in X.columns and 'Rainfall' in X.columns:
            X.loc[(X["RainToday"].isna()) & (X["Rainfall"] >= 1), "RainToday"] = 1
            X.loc[(X["RainToday"].isna()) & (X["Rainfall"] >= 0) & (X["Rainfall"] < 1), "RainToday"] = 0

        # Supprimer les lignes restantes avec NaN dans Rainfall ou RainToday
        X.dropna(subset=["Rainfall", "RainToday"], inplace=True)

        return X

    def create_distance_matrix(self):
        # Coordonnées des stations

        cities_coordinates = {
            'Albury': (-36.0734, 146.9169),
            'BadgerysCreek': (-33.8891, 150.7725),
            'Cobar': (-31.8647, 145.8225),
            'CoffsHarbour': (-30.2989, 153.1145),
            'Moree': (-29.4736, 149.8411),
            'Newcastle': (-32.9283, 151.7817),
            'NorahHead': (-33.1920, 151.5216),
            'NorfolkIsland': (-29.0408, 167.9591),
            'Penrith': (-33.7491, 150.6941),
            'Richmond': (-33.5990, 150.7670),
            'Sydney': (-33.8688, 151.2093),
            'SydneyAirport': (-33.9399, 151.1753),
            'WaggaWagga': (-35.1100, 147.3673),
            'Williamtown': (-32.7990, 151.8430),
            'Wollongong': (-34.4278, 150.8931),
            'Canberra': (-35.2809, 149.1300),
            'Tuggeranong': (-35.4135, 149.0705),
            'MountGinini': (-35.4800, 148.9325),
            'Ballarat': (-37.5622, 143.8503),
            'Bendigo': (-36.7580, 144.2805),
            'Sale': (-38.1000, 147.0667),
            'MelbourneAirport': (-37.6733, 144.8431),
            'Melbourne': (-37.8136, 144.9631),
            'Mildura': (-34.1842, 142.1593),
            'Nhil': (-35.6544, 141.6931),
            'Portland': (-38.3553, 141.5883),
            'Watsonia': (-37.7170, 145.1265),
            'Dartmoor': (-37.8689, 141.4203),
            'Brisbane': (-27.4698, 153.0251),
            'Cairns': (-16.9203, 145.7710),
            'GoldCoast': (-28.0167, 153.4000),
            'Townsville': (-19.2589, 146.8184),
            'Adelaide': (-34.9285, 138.6007),
            'MountGambier': (-37.8333, 140.7833),
            'Nuriootpa': (-34.4934, 138.9773),
            'Woomera': (-31.1333, 136.8333),
            'Albany': (-35.0200, 117.8833),
            'Witchcliffe': (-33.7969, 115.1236),
            'PearceRAAF': (-31.8036, 115.9747),
            'PerthAirport': (-31.9385, 115.9773),
            'Perth': (-31.9505, 115.8605),
            'SalmonGums': (-33.0091, 120.3702),
            'Walpole': (-34.9789, 116.6293),
            'Hobart': (-42.8821, 147.3272),
            'Launceston': (-41.4403, 147.1349),
            'AliceSprings': (-23.6980, 133.8807),
            'Darwin': (-12.4634, 130.8456),
            'Katherine': (-14.4683, 132.2615),
            'Uluru': (-25.3444, 131.0369)
        }

        location_list = list(cities_coordinates.keys())
        dist_mat = pd.DataFrame(index=location_list)

        for location in location_list:
            for location2 in location_list:
                dist_mat.loc[location, location2] = \
                    geopy.distance.distance(cities_coordinates[location],
                                           cities_coordinates[location2]).km

        return dist_mat

    def complete_na_neareststation(self, df, variable, distance_max):
        dist_mat = self.create_distance_matrix()

        # Récupérer les lignes où la variable est manquante
        df_na_variable = df[(df[variable].isna())][["Date", "Location", variable]]
        location_list = df_na_variable.Location.unique()
        df_na_variable[variable + "_near"] = np.nan

        total_count = 0

        for location in location_list:
            # print("Station :", location)
            # Stations à moins de distance_max km
            nearest_stations = dist_mat.loc[location, dist_mat.loc[location] < distance_max]
            nearest_stations = nearest_stations[nearest_stations.index != location]
            count = 0

            if len(nearest_stations) == 0:
                print(f"Pas de station à moins de {distance_max} km de {location}")
            else:
                # On garde la plus proche
                station_check = nearest_stations.sort_values().index[0]
                dates_to_check = list(df_na_variable.loc[(df_na_variable["Location"] == location), "Date"])
                # print(f"Station la plus proche : {station_check}")

                # boucles sur les dates Nas pour récupérer les valeurs de la station à proximité
                for date_tmp in dates_to_check:
                    # print(date_tmp)
                    # On ne teste que les dates qui existent aussi pour la station la plus proche
                    if isinstance(df.index, pd.MultiIndex):
                        # Si MultiIndex, on récupère le deuxième niveau (Date)
                        if date_tmp in df.loc[df["Location"] == station_check].index.get_level_values(1):
                            # Si une valeur existe, on l'impute
                            if df.loc[(station_check, date_tmp), variable] != np.nan:
                                df_na_variable.loc[(location, date_tmp), variable + "_near"] = \
                                    df.loc[(station_check, date_tmp), variable]

                                count += 1
                    else:
                        # Si pas de MultiIndex, vérifier simplement avec l'index "Date"
                        if date_tmp in df[df["Location"] == station_check].index:
                            if df.loc[date_tmp, variable] != np.nan:
                                df_na_variable.loc[(location, date_tmp), variable + "_near"] = \
                                    df.loc[date_tmp, variable]
                                count += 1

            # print(f"{count} valeurs récupérées pour la variable {variable} et la station {location}")
            total_count += count

        # print(f"{total_count} valeurs récupérées")
        return df_na_variable
