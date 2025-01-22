# Importer les bibliothèques
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from sklearn.model_selection import train_test_split, ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
import os
import sys
import warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore", ConvergenceWarning)


########################################################################################################

## Lecture du dataframe & Sélection des données

# Obtenir le chemin du dossier 
base_dir = Path(__file__).resolve().parent
data_path = base_dir / "df_final.csv"

# Chargement des données
df_final = pd.read_csv(data_path)
df_final = df_final.rename(columns={"id_Location": "Location", "id_Date": "Date"})
df_final["Date"] = pd.to_datetime(df_final["Date"])
df_final = df_final.drop(columns=["Climate_Desert", "Climate_Grassland", "Climate_Subtropical", "Climate_Temperate", "Climate_Tropical",])
df_final = df_final.sort_values(by="Date", ascending=True)

# Grouper les données par Location
output_location = base_dir / "data_location"
output_location = output_location.resolve()
if not output_location.exists():
    output_location.mkdir(parents=True)
groupes = df_final.groupby("Location")
for location, groupe in groupes:
    nouveau_df = groupe.copy()
    output_path = os.path.join(output_location, f"df_{location}.csv")
    nouveau_df.to_csv(output_path, index=False)

### J'ai essayer de faire sans créer de fichiers csv intermédiaires, mais le reste du code si je ne fait pas ça !!!!
### J'ai essayé de stocker les données dans un dictionnaire mais ça génère des erreurs :
### ValueError: Provided exogenous values are not of the appropriate shape. Required (668, 25), got (2669, 25)

########################################################################################################

## Sélection de la Sation/Ville étudiée

# Location étudiée
station_name = "Sydney"  ### !!! Attention à bien changer le nom de la ville ici !!! ###

# Chargement des données
df_location_path = Path(__file__).resolve().parent / "data_location" / f"df_{station_name}.csv"
df_location = pd.read_csv(df_location_path)
df_location["Date"] = pd.to_datetime(df_location["Date"])

# Infos sur le dataframe
print(f"\nInfos sur df_location - {station_name} :\n")
print(df_location.info(), "\n")
print(f"\nHead df_location - {station_name} :\n")
print(df_location.head(), "n")

# Créer un dossier pour sauvegarder les graphes/fichiers des modèles
output_model = base_dir / "modeling_MaxTemp_MinTemp_results" / f"{station_name}"
output_model = output_model.resolve()
if not output_model.exists():
    output_model.mkdir(parents=True)

########################################################################################################

## Variable MaxTemp

# Modèle SARIMA MaxTemp
def model_sarima_MaxTemp(df, station_name, output_model):
    """
    Analyse et modélisation SARIMA pour une série temporelle donnée.

    Args:
        df (pd.DataFrame): DataFrame contenant les données avec les colonnes ["Date", "MaxTemp", ...].
        station_name (str): Nom de la station pour les visualisations et fichiers.
        output_model (str): Répertoire où sauvegarder les fichiers de sortie.

    Returns:
        dict: Résultats d'évaluation du modèle (MSE, RMSE, MAE, R2, MAPE).
    """
    # Assurez-vous que le répertoire de sortie existe
    os.makedirs(output_model, exist_ok=True)

    # Visualisation de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.plot(df["Date"], df["MaxTemp"], linewidth=0.8, label="Température maximale", color="firebrick") 
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Température maximale (°C)")
    plt.title(f"MaxTemp {station_name} - SARIMA : Évolution de la variable", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"1. {station_name}_MaxTemp_Évolution.png"))
    plt.close()

    # Décomposition de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", ["red", "red", "red", "red"])})
    result = seasonal_decompose(df["MaxTemp"], model="additive", period=365)
    fig = result.plot()
    fig.axes[0].plot(result.observed, color="firebrick", linewidth=0.5)
    fig.axes[1].plot(result.trend, color="firebrick", linewidth=0.5)
    fig.axes[2].plot(result.seasonal, color="firebrick", linewidth=0.5)
    fig.axes[3].scatter(result.resid.index, result.resid, color="firebrick")
    for ax in fig.axes:
        ax.set_title("") # Enlever les titres des sous-graphiques
    fig.suptitle(f"MaxTemp {station_name} - SARIMA : Décomposition de la série temporelle", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"2. {station_name}_MaxTemp_TimeSeries_Décomposition.png"))
    plt.show()  # plt.show() au lieu de plt.close() pour ajuster la taille et enregistrer manuellement, sinon le graph est compressé

    # Diviser les données en ensemble d'entraînement et de test
    train = df["MaxTemp"][:int(0.8 * len(df))]
    test = df["MaxTemp"][int(0.8 * len(df)):]

    # Enregistement des données exogènes et synchronisation des indices
    train_features = df.drop(columns=["Date", "Location", "MaxTemp"]).iloc[:int(0.8 * len(df))]
    test_features = df.drop(columns=["Date", "Location", "MaxTemp"]).iloc[int(0.8 * len(df)):]
    train_features = train_features.to_numpy()
    test_features = test_features.to_numpy()
    
    # Affichage ACF et PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plot_acf(train, lags=50, ax=ax1, color="firebrick")
    plot_pacf(train, lags=50, ax=ax2, color="firebrick")
    ax1.set_title(f"MaxTemp {station_name} - SARIMA : ACF", fontsize=15)
    ax2.set_title(f"MaxTemp {station_name} - SARIMA : PACF", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"3. {station_name}_MaxTemp_ACF_PACF.png"))
    plt.close()

    # Création et ajustement du modèle SARIMA
    sarima_model = SARIMAX(train, 
                           exog=train_features, 
                           order=(1, 1, 1), 
                           seasonal_order=(1, 1, 1, 12), 
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
    sarima_results = sarima_model.fit()
    
    # Sauvegarder le résumé du modèle
    results_file = os.path.join(output_model, f"4. {station_name}_MaxTemp_SARIMA_Results.txt")
    with open(results_file, "w") as f:
        f.write(sarima_results.summary().as_text())

    # Prédictions du modèle SARIMA
    predictions = sarima_results.predict(start=test.index[0], 
                                         end=test.index[-1], 
                                         exog=test_features, 
                                         dynamic=False)

    # Visualisation des prédictions
    plt.figure(figsize=(15, 7))
    plt.plot(train, label="Train", color="black", linewidth=0.8)
    plt.plot(test, label="Test", color="gray", linewidth=0.8)
    plt.plot(predictions, label="Prédictions", color="red", linestyle="dotted")
    plt.title(f"MaxTemp {station_name} - SARIMA : Prédictions du modèle", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("Température maximale (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"5. {station_name}_MaxTemp_SARIMA_Prédictions.png"))
    plt.close()

    # Évaluation du modèle
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    evaluation_file = os.path.join(output_model, f"6. {station_name}_MaxTemp_Évaluation_modèle.txt")
    with open(evaluation_file, "w") as f:
        f.write(f"Evaluation du modèle SARIMA pour MaxTemp {station_name} \n\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"R-squared: {r2}\n")
        f.write(f"Mean Absolute Percentage Error: {mape}%\n")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# Modèle Prophet MaxTemp
def model_prophet_Maxtemp(df_location, station_name, output_model):
    """
    Fonction pour modéliser et prédire les températures maximales avec Prophet.

    Args:
        df_location (DataFrame): Données contenant les colonnes "Date" et "MaxTemp".
        station_name (str): Nom de la station pour personnaliser les visualisations et les fichiers.
        output_model (str): Répertoire de sauvegarde des résultats.

    Returns:
        None
    """
    # Charger et préparer les données pour Prophet
    prophet_df = df_location[["Date", "MaxTemp"]].rename(columns={"Date": "ds", "MaxTemp": "y"})

    # Créer et configurer le modèle Prophet
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                            daily_seasonality=False, seasonality_mode="additive")

    # Diviser les données en ensemble d'entraînement et de test
    train_prophet = prophet_df[:int(0.8 * len(prophet_df))]  # 80% pour l'entraînement
    test_prophet = prophet_df[int(0.8 * len(prophet_df)):]   # 20% pour le test
    prophet_model.fit(train_prophet)

    # Générer des prédictions sur la période de test
    future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq="D")
    forecast = prophet_model.predict(future)

    # Visualisation des composants des prédictions
    fig_components = prophet_model.plot_components(forecast)
    axes = fig_components.get_axes()
    for ax in axes:
        ax.set_title(f"MaxTemp {station_name} - Prophet : Composants des prédictions", fontsize=13)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    for ax in axes:
        for line in ax.get_lines():
            line.set_color("firebrick")
        for collection in ax.collections:
            collection.set_facecolor("red")
            collection.set_alpha(0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"7. {station_name}_MaxTemp_Prophet_Components.png"))
    plt.close()

    # Visualisation des prédictions
    fig = prophet_model.plot(forecast)
    ax = fig.gca()
    for line in ax.get_lines():
        line.set_color("firebrick")
    ax.collections[0].set_facecolor("red")
    ax.collections[0].set_alpha(0.3)
    plt.title(f"MaxTemp {station_name} - Prophet : Prédictions", fontsize=15, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Température maximale (°C)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"8. {station_name}_MaxTemp_Prophet_Predictions.png"))
    plt.close()

    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(15, 7))
    plt.plot(test_prophet["ds"], test_prophet["y"], label="Valeurs réelles", color="firebrick", linewidth=1)
    plt.plot(test_prophet["ds"], forecast["yhat"].tail(len(test_prophet)), label="Prédictions", color="gray", linestyle="--", linewidth=1)
    plt.fill_between(test_prophet["ds"], forecast["yhat_lower"].tail(len(test_prophet)), forecast["yhat_upper"].tail(len(test_prophet)),
                     color="gray", alpha=0.2, label="Intervalle de confiance (80%)")
    plt.title(f"MaxTemp {station_name} - Prophet : Prédictions vs. Valeurs réelles", fontsize=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Température maximale (°C)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"9. {station_name}_MaxTemp_Prophet_Predictions_vs_Real.png"))
    plt.close()

    # Évaluation des performances du modèle Prophet
    y_true = test_prophet["y"].values
    y_pred = forecast.loc[test_prophet.index, "yhat"].values
    mse_prophet = mean_squared_error(y_true, y_pred)
    rmse_prophet = np.sqrt(mse_prophet)
    mae_prophet = mean_absolute_error(y_true, y_pred)
    r2_prophet = r2_score(y_true, y_pred)
    mape_prophet = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Sauvegarder l'évaluation dans un fichier texte
    evaluation_file_prophet = os.path.join(output_model, f"10. {station_name}_MaxTemp_Prophet_Évaluation_modèle.txt")
    with open(evaluation_file_prophet, "w") as f:
        f.write(f"Evaluation du modèle Prophet pour MaxTemp {station_name} \n\n")
        f.write(f"Mean Squared Error: {mse_prophet}\n")
        f.write(f"Root Mean Squared Error: {rmse_prophet}\n")
        f.write(f"Mean Absolute Error: {mae_prophet}\n")
        f.write(f"R-squared: {r2_prophet}\n")
        f.write(f"Mean Absolute Percentage Error: {mape_prophet}%\n")

# Appeler les fonctions de modeling
model_sarima_MaxTemp(df_location, station_name, output_model)
model_prophet_Maxtemp(df_location, station_name, output_model)


########################################################################################################

## Variable MinTemp

# Modèle SARIMA MinTemp
def model_sarima_MinTemp(df, station_name, output_model):
    """
    Analyse et modélisation SARIMA pour une série temporelle donnée.

    Args:
        df (pd.DataFrame): DataFrame contenant les données avec les colonnes ["Date", "MinTemp", ...].
        station_name (str): Nom de la station pour les visualisations et fichiers.
        output_model (str): Répertoire où sauvegarder les fichiers de sortie.

    Returns:
        dict: Résultats d'évaluation du modèle (MSE, RMSE, MAE, R2, MAPE).
    """
    # Assurez-vous que le répertoire de sortie existe
    os.makedirs(output_model, exist_ok=True)

    # Visualisation de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.plot(df["Date"], df["MinTemp"], linewidth=0.8, label="Température minimale", color="navy") 
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Température minimale (°C)")
    plt.title(f"MinTemp {station_name} - SARIMA : Évolution de la variable", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"1. {station_name}_MinTemp_Évolution.png"))
    plt.close()

    # Décomposition de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", ["royalblue", "royalblue", "royalblue", "royalblue"])})
    result = seasonal_decompose(df["MinTemp"], model="additive", period=365)
    fig = result.plot()
    fig.axes[0].plot(result.observed, color="navy", linewidth=0.5)
    fig.axes[1].plot(result.trend, color="navy", linewidth=0.5)
    fig.axes[2].plot(result.seasonal, color="navy", linewidth=0.5)
    fig.axes[3].scatter(result.resid.index, result.resid, color="navy")
    for ax in fig.axes:
        ax.set_title("") # Enlever les titres des sous-graphiques
    fig.suptitle(f"MinTemp {station_name} - SARIMA : Décomposition de la série temporelle", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"2. {station_name}_MinTemp_TimeSeries_Décomposition.png"))
    plt.show()  # plt.show() au lieu de plt.close() pour ajuster la taille et enregistrer manuellement, sinon le graph est compressé

    # Diviser les données en ensemble d'entraînement et de test
    train = df["MinTemp"][:int(0.8 * len(df))]
    test = df["MinTemp"][int(0.8 * len(df)):]

    # Enregistement des données exogènes et synchronisation des indices
    train_features = df.drop(columns=["Date", "Location", "MinTemp"]).iloc[:int(0.8 * len(df))]
    test_features = df.drop(columns=["Date", "Location", "MinTemp"]).iloc[int(0.8 * len(df)):]
    train_features = train_features.to_numpy()
    test_features = test_features.to_numpy()
    
    # Affichage ACF et PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plot_acf(train, lags=50, ax=ax1, color="navy")
    plot_pacf(train, lags=50, ax=ax2, color="navy")
    ax1.set_title(f"MinTemp {station_name} - SARIMA : ACF", fontsize=15)
    ax2.set_title(f"MinTemp {station_name} - SARIMA : PACF", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"3. {station_name}_MinTemp_ACF_PACF.png"))
    plt.close()

    # Création et ajustement du modèle SARIMA
    sarima_model = SARIMAX(train, 
                           exog=train_features, 
                           order=(1, 1, 1), 
                           seasonal_order=(1, 1, 1, 12), 
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
    sarima_results = sarima_model.fit()
    
    # Sauvegarder le résumé du modèle
    results_file = os.path.join(output_model, f"4. {station_name}_MinTemp_SARIMA_Results.txt")
    with open(results_file, "w") as f:
        f.write(sarima_results.summary().as_text())

    # Prédictions du modèle SARIMA
    predictions = sarima_results.predict(start=test.index[0], 
                                         end=test.index[-1], 
                                         exog=test_features, 
                                         dynamic=False)

    # Visualisation des prédictions
    plt.figure(figsize=(15, 7))
    plt.plot(train, label="Train", color="black", linewidth=0.8)
    plt.plot(test, label="Test", color="gray", linewidth=0.8)
    plt.plot(predictions, label="Prédictions", color="red", linestyle="dotted")
    plt.title(f"MinTemp {station_name} - SARIMA : Prédictions du modèle", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("Température minimale (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"5. {station_name}_MinTemp_SARIMA_Prédictions.png"))
    plt.close()

    # Évaluation du modèle
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    evaluation_file = os.path.join(output_model, f"6. {station_name}_MinTemp_Évaluation_modèle.txt")
    with open(evaluation_file, "w") as f:
        f.write(f"Evaluation du modèle SARIMA pour MinTemp {station_name} \n\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"Mean Absolute Error: {mae}\n")
        f.write(f"R-squared: {r2}\n")
        f.write(f"Mean Absolute Percentage Error: {mape}%\n")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# Modèle Prophet MinTemp
def model_prophet_MinTemp(df_location, station_name, output_model):
    """
    Fonction pour modéliser et prédire les températures minimales avec Prophet.

    Args:
        df_location (DataFrame): Données contenant les colonnes "Date" et "MinTemp".
        station_name (str): Nom de la station pour personnaliser les visualisations et les fichiers.
        output_model (str): Répertoire de sauvegarde des résultats.

    Returns:
        None
    """
    # Charger et préparer les données pour Prophet
    prophet_df = df_location[["Date", "MinTemp"]].rename(columns={"Date": "ds", "MinTemp": "y"})

    # Créer et configurer le modèle Prophet
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                            daily_seasonality=False, seasonality_mode="additive")

    # Diviser les données en ensemble d'entraînement et de test
    train_prophet = prophet_df[:int(0.8 * len(prophet_df))]  # 80% pour l'entraînement
    test_prophet = prophet_df[int(0.8 * len(prophet_df)):]   # 20% pour le test
    prophet_model.fit(train_prophet)

    # Générer des prédictions sur la période de test
    future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq="D")
    forecast = prophet_model.predict(future)

    # Visualisation des composants des prédictions
    fig_components = prophet_model.plot_components(forecast)
    axes = fig_components.get_axes()
    for ax in axes:
        ax.set_title(f"MinTemp {station_name} - Prophet : Composants des prédictions", fontsize=13)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    for ax in axes:
        for line in ax.get_lines():
            line.set_color("navy")
        for collection in ax.collections:
            collection.set_facecolor("royalblue")
            collection.set_alpha(0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"7. {station_name}_MinTemp_Prophet_Components.png"))
    plt.close()

    # Visualisation des prédictions
    fig = prophet_model.plot(forecast)
    ax = fig.gca()
    for line in ax.get_lines():
        line.set_color("navy")
    ax.collections[0].set_facecolor("royalblue")
    ax.collections[0].set_alpha(0.3)
    plt.title(f"MinTemp {station_name} - Prophet : Prédictions", fontsize=15, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Température minimale (°C)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"8. {station_name}_MinTemp_Prophet_Predictions.png"))
    plt.close()

    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(15, 7))
    plt.plot(test_prophet["ds"], test_prophet["y"], label="Valeurs réelles", color="navy", linewidth=1)
    plt.plot(test_prophet["ds"], forecast["yhat"].tail(len(test_prophet)), label="Prédictions", color="gray", linestyle="--", linewidth=1)
    plt.fill_between(test_prophet["ds"], forecast["yhat_lower"].tail(len(test_prophet)), forecast["yhat_upper"].tail(len(test_prophet)),
                     color="gray", alpha=0.2, label="Intervalle de confiance (80%)")
    plt.title(f"MinTemp {station_name} - Prophet : Prédictions vs. Valeurs réelles", fontsize=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Température minimale (°C)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"9. {station_name}_MinTemp_Prophet_Predictions_vs_Real.png"))
    plt.close()

    # Évaluation des performances du modèle Prophet
    y_true = test_prophet["y"].values
    y_pred = forecast.loc[test_prophet.index, "yhat"].values
    mse_prophet = mean_squared_error(y_true, y_pred)
    rmse_prophet = np.sqrt(mse_prophet)
    mae_prophet = mean_absolute_error(y_true, y_pred)
    r2_prophet = r2_score(y_true, y_pred)
    mape_prophet = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Sauvegarder l'évaluation dans un fichier texte
    evaluation_file_prophet = os.path.join(output_model, f"10. {station_name}_MinTemp_Prophet_Évaluation_modèle.txt")
    with open(evaluation_file_prophet, "w") as f:
        f.write(f"Evaluation du modèle Prophet pour MinTemp {station_name} \n\n")
        f.write(f"Mean Squared Error: {mse_prophet}\n")
        f.write(f"Root Mean Squared Error: {rmse_prophet}\n")
        f.write(f"Mean Absolute Error: {mae_prophet}\n")
        f.write(f"R-squared: {r2_prophet}\n")
        f.write(f"Mean Absolute Percentage Error: {mape_prophet}%\n")

# Appeler les fonctions de modeling
model_sarima_MinTemp(df_location, station_name, output_model)
model_prophet_MinTemp(df_location, station_name, output_model)


########################################################################################################
