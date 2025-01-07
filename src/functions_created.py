import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats  import chi2_contingency
from datetime import timedelta

# Ce script comporte différentes fonctions annexes utiles 

# Fonction qui crée la liste des dates sur la période d'étude

def create_date_list(df):
    # création de la liste des dates sur l'ensemble de la période ####
    
    # Define start and end dates
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    start_date
    # Initialize an empty list
    date_list = []

    # Loop through the range of dates and append to the list
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += timedelta(days=1)

    df_dates = pd.DataFrame({"Dates" : date_list})
    df_dates["Year"] = df_dates["Dates"].dt.year
    df_dates["Year"].value_counts()
    df_dates["Location"] = "Complete serie"

    return df_dates, date_list


import geopy.distance 

def create_distance_matrix():
    # coordonnées des stations

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
    dist_mat = pd.DataFrame(index = location_list)
    
    for location in location_list:
        for location2 in location_list:
            dist_mat.loc[location, location2] = \
                geopy.distance.distance(cities_coordinates[location], 
                                        cities_coordinates[location2]).km
            
        
    return dist_mat

# Fonction qui complète une variable à partir des données de la station la plus proche
# distance_max : distance maximale pour laquelle on recherche une station en km
# Le Na est remplacé si il existe une valeur pour la même date dans la station la plus proche

def complete_na_neareststation(df, variable, distance_max):
    dist_mat = create_distance_matrix()
   
    #On récupère les dates Nas pour la variable
    df_na_variable = df[(df[variable].isna())][["Date", "Location", variable]]
    location_list = df_na_variable.Location.unique()
    df_na_variable[variable + "_near"] = np.nan

    total_count = 0

    for location in location_list:
        print("Station :", location)
        # Stations à moins de  distance_max km
        nearest_stations = dist_mat.loc[location, dist_mat.loc[location] <   distance_max]
        nearest_stations = nearest_stations[nearest_stations.index != location]
        count = 0

        if len(nearest_stations) == 0:
            print("Pas de station à moins de ", distance_max, " km de ", location)

        else:
            # On garde la plus proche 
            station_check = nearest_stations.sort_values().index[0]
            dates_to_check = list(df_na_variable.loc[(df_na_variable["Location"] == location),"Date"])
            print("Station la plus proche : ", station_check)
            
            # boucles sur les dates Nas pour récupérer les valeurs de la station à proximité
            for date_tmp in dates_to_check:
                # print(date_tmp)
                # On ne teste que les dates qui existent aussi pour la station la plus proche
                if date_tmp in df.loc[df["Location"] == station_check].index.get_level_values(1):
                    # Si une valeur existe, on l'impute
                    if df.loc[(station_check, date_tmp), variable] != np.nan:
                        df_na_variable.loc[(location, date_tmp), variable + "_near"] = \
                            df.loc[(station_check, date_tmp), variable]
                        
                        count += 1
            
        print(count, " valeurs récupérées pour la variable ", variable, "et la station", location)
        total_count += count
    
    print(total_count, "Valeurs récupérées")
    return df_na_variable

# cercle de corrélation d'une PCA pour deux axes et toutes les features d'un df

def draw_correlation_circle(df_charges_factorielles, 
                            pca, 
                            arrow_length=0.1, 
                            label_rotation=0, 
                            Axes_names=['Axe 1', 'Axe 2']):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, var in enumerate(df_charges_factorielles.columns):
        x = df_charges_factorielles.loc[Axes_names[0], var]
        y = df_charges_factorielles.loc[Axes_names[1], var]
        ax.arrow(0, 0, x, y, head_width=arrow_length, 
                 head_length=arrow_length, fc='gray', ec='gray')
        ax.text(x*1.15, y*1.15, var, ha='center', va='center', 
                rotation=label_rotation, fontsize=9)
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black')
    ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(Axes_names[0])
    ax.set_ylabel(Axes_names[1])
    ax.set_title('Cercle des Corrélations')
    plt.grid()
    plt.show()


# Analyse de la distribution d'une variable binaire en fonction de la target

def analyse_variable_binaire(variable_name, target_name, base):

    print("Répartition de la variable: ", end="\n\n")

    print(base[variable_name].value_counts(normalize=True, dropna=False),
          end="\n\n")

    print("Matrice de contingence: ", end="\n\n")

    print(pd.crosstab(base[variable_name], base[target_name], normalize='index'))

    sns.countplot(x=variable_name, data=base)

    plt.title(f'Répartition de {variable_name} \n', fontsize=20)

    plt.show()

# Analyse de la distribution d'une variable quantitative

def analyse_variable_quantitative(variable_name, base):

    print()

    print("Statistiques de la variable: ", end="\n\n")

    print(base[variable_name].describe(), end="\n\n")

    sns.boxplot(x=variable_name, data=base)

    plt.title(f'Distribution de {variable_name} \n', fontsize=20)

    plt.show()

    sns.histplot(x=variable_name, data=base,)

    plt.title(f'Distribution de {variable_name} \n', fontsize=20)

    plt.show()

# Test de Chi2 entre deux variables qualitatives et calcul du V_cramer

def chi2_test(variable1, variable2, base):

    # Stat et p_value du test
    stat, p = chi2_contingency(pd.crosstab(
    base[variable1], base[variable2]))[:2]
    # V de Cramer
    V_Cramer = np.sqrt(stat/pd.crosstab( base[variable1],base[variable2]).values.sum())
    print()
    print("Le V de cramer est égal à : ", V_Cramer, end="\n\n")
    print("La p-valeur du test de Chi-Deux est égal à : ", p)

    return stat, p, V_Cramer

