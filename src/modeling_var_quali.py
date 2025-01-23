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

base_dir = Path(__file__).resolve().parent

########################################################################################################

# Sélection de la Location
def location_selection(station_name:str, base_dir:Path):
    """
    Cette fonction charge les données pour une station spécifique, 
    affiche des informations sur le dataframe et crée un dossier 
    pour sauvegarder les résultats des modèles.

    Args:
    - station_name (str): Le nom de la station à analyser.
    - base_dir (Path): Le répertoire de base où enregistrer les résultats.
    """
    # Chargement des données
    df_location_path = base_dir / "data_location" / f"df_{station_name}.csv"
    df_location = pd.read_csv(df_location_path)
    df_location = df_location.rename(columns={"id_Location": "Location", "id_Date": "Date"})
    df_location["Date"] = pd.to_datetime(df_location["Date"])
    df_location = df_location.sort_values(by="Date", ascending=True)

    # Transformation des variables
    numeric_cols = df_location.select_dtypes(include=["bool"]).columns
    df_location[numeric_cols] = df_location[numeric_cols].astype(int)

    # Infos sur le dataframe
    print(f"\nInfos sur df_location - {station_name} :\n")
    print(df_location.info(), "\n")
    print(f"\nHead df_location - {station_name} :\n")
    print(df_location.head(), "\n")

    # Créer un dossier pour sauvegarder les graphes/fichiers des modèles
    output_model = base_dir / "modeling_MaxTemp_MinTemp_results" / f"{station_name}"
    output_model = output_model.resolve()
    if not output_model.exists():
        output_model.mkdir(parents=True)

    return df_location, output_model

# Modèle SARIMA
def model_sarima(df, station_name, variable_name, output_model):
    """
    Analyse et modélisation SARIMA pour une série temporelle donnée.

    Args:
        df (pd.DataFrame): DataFrame contenant les données avec les colonnes ["Date", "MaxTemp", ...].
        station_name (str): Nom de la station pour les visualisations et fichiers.
        output_model (str): Répertoire où sauvegarder les fichiers de sortie.
        variable_name (str): Nom de la variable à modéliser, par défaut "MaxTemp".

    Returns:
        dict: Résultats d'évaluation du modèle (MSE, RMSE, MAE, R2, MAPE).
    """
    # Assurez-vous que le répertoire de sortie existe
    os.makedirs(output_model, exist_ok=True)

    # Visualisation de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.plot(df["Date"], df[variable_name], linewidth=0.8, label=variable_name, color="navy")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"{variable_name}")
    plt.title(f"{variable_name} {station_name} - SARIMA : Évolution de la variable", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"1.{station_name}_{variable_name}_Évolution.png"))
    plt.close()

    # Décomposition de la série temporelle
    plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", ["green", "green", "green", "green"])})
    result = seasonal_decompose(df[variable_name], model="additive", period=365)
    fig = result.plot()
    fig.set_size_inches(20, 10)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.set_dpi(500)
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(output_model, f"2.{station_name}_{variable_name}_TimeSeries_Décomposition.png"))
    plt.close()

    # Diviser les données en ensemble d'entraînement et de test
    train = df[variable_name][:int(0.8 * len(df))]
    test = df[variable_name][int(0.8 * len(df)):]

    # Enregistrement des données exogènes et synchronisation des indices
    train_features = df.drop(columns=["Date", "Location", variable_name]).iloc[:int(0.8 * len(df))]
    test_features = df.drop(columns=["Date", "Location", variable_name]).iloc[int(0.8 * len(df)):]
    train_features = train_features.to_numpy()
    test_features = test_features.to_numpy()

    # Affichage ACF et PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plot_acf(train, lags=50, ax=ax1, color="green")
    plot_pacf(train, lags=50, ax=ax2, color="green")
    ax1.set_title(f"{variable_name} {station_name} - SARIMA : ACF", fontsize=15)
    ax2.set_title(f"{variable_name} {station_name} - SARIMA : PACF", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"3.{station_name}_{variable_name}_ACF_PACF.png"))
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
    results_file = os.path.join(output_model, f"4.{station_name}_{variable_name}_SARIMA_Results.txt")
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
    plt.title(f"{variable_name} {station_name} - SARIMA : Prédictions du modèle", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel(f"{variable_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"5.{station_name}_{variable_name}_SARIMA_Prédictions.png"))
    plt.close()

    # Évaluation du modèle
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    evaluation_file = os.path.join(output_model, f"6.{station_name}_{variable_name}_Évaluation_modèle.txt")
    with open(evaluation_file, "w") as f:
        f.write(f"Evaluation du modèle SARIMA pour {variable_name} {station_name} \n\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"MAPE: {mape}%\n")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# Modèle Prophet
def model_prophet(df, station_name, variable_name, output_model):
    """
    Fonction pour modéliser et prédire les températures maximales avec Prophet.

    Args:
        df_location (pd.DataFrame): DataFrame avec les données de la station.
        station_name (str): Le nom de la station pour l'enregistrement.
        output_model (str): Répertoire de sortie pour les résultats.
        variable_name (str): Nom de la variable à modéliser.

    Returns:
        dict: Métriques de performance pour le modèle.
    """
    # Charger et préparer les données pour Prophet
    prophet_df = df_location[["Date", variable_name]].rename(columns={"Date": "ds", variable_name: "y"})

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
        ax.set_title(f"{variable_name} {station_name} - Prophet : Composants des prédictions", fontsize=13)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    for ax in axes:
        for line in ax.get_lines():
            line.set_color("green")
        for collection in ax.collections:
            collection.set_facecolor("green")
            collection.set_alpha(0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"7.{station_name}_{variable_name}_Prophet_Components.png"))
    plt.close()

    # Visualisation des prédictions
    fig = prophet_model.plot(forecast)
    ax = fig.gca()
    for line in ax.get_lines():
        line.set_color("gray")
    ax.collections[0].set_facecolor("red")
    ax.collections[0].set_alpha(0.3)
    plt.title(f"{variable_name} {station_name} - Prophet : Prédictions", fontsize=15, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{variable_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"8.{station_name}_{variable_name}_Prophet_Predictions.png"))
    plt.close()

    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(15, 7))
    plt.plot(test_prophet["ds"], test_prophet["y"], label="Valeurs réelles", color="gray", linewidth=1)
    plt.plot(test_prophet["ds"], forecast["yhat"].tail(len(test_prophet)), label="Prédictions", color="red", linestyle="--", linewidth=1)
    plt.fill_between(test_prophet["ds"], forecast["yhat_lower"].tail(len(test_prophet)), forecast["yhat_upper"].tail(len(test_prophet)), color="gray", alpha=0.2)
    plt.title(f"{variable_name} {station_name} - Prophet : Comparaison des prédictions avec les valeurs réelles", fontsize=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{variable_name}", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"9.{station_name}_{variable_name}_Prophet_Predictions_vs_Actual.png"))
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
    evaluation_file_prophet = os.path.join(output_model, f"10.{station_name}_{variable_name}_Prophet_Évaluation_modèle.txt")
    with open(evaluation_file_prophet, "w") as f:
        f.write(f"Evaluation du modèle Prophet pour {variable_name} {station_name} \n\n")
        f.write(f"MSE: {mse_prophet}\n")
        f.write(f"REMSE: {rmse_prophet}\n")
        f.write(f"MAE: {mae_prophet}\n")
        f.write(f"R2: {r2_prophet}\n")
        f.write(f"MAPE: {mape_prophet}%\n")

    
########################################################################################################

## Sélection des Location étudiées

variable_name = "MaxTemp"
station_names = ["Sydney"]
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "MinTemp"
station_names = ["Sydney", "Adelaide", "AliceSprings", "Brisbane", "Cairns", "Canberra",
                 "Darwin", "Hobart", "Melbourne", "Perth", "Uluru"]
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)
