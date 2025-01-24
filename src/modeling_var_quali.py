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
    df_location_path = base_dir / "data_location_V2" / f"df_{station_name}.csv"
    df_location = pd.read_csv(df_location_path)
    df_location["id_Date"] = pd.to_datetime(df_location["id_Date"])
    df_location = df_location.sort_values(by="id_Date", ascending=True)
    df_location = df_location.loc[:, ~df_location.columns.str.contains("Year", case=False)]
    
    # Infos sur le dataframe
    print(f"\nInfos sur df_location - {station_name} :\n")
    print(df_location.info(), "\n")
    print(f"\nHead df_location - {station_name} :\n")
    print(df_location.head(), "\n")

    # Créer un dossier pour sauvegarder les graphes/fichiers des modèles
    output_model = base_dir / "times_series_modeling_results" / f"{variable_name}" / f"{station_name}"
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
    # Vérifier que le répertoire de sortie existe
    os.makedirs(output_model, exist_ok=True)

    # Visualisation de la série temporelle
    plt.figure(figsize=(15, 7))
    plt.plot(df["id_Date"], df[variable_name], linewidth=0.8, label=variable_name, color="navy")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(f"{variable_name}")
    plt.title(f"{variable_name} {station_name} - SARIMA : Évolution de la variable", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_Évolution.png"))
    plt.close()

    # Décomposition de la série temporelle
    plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", ["green", "green", "green", "green"])})
    result = seasonal_decompose(df[variable_name], model="additive", period=365)
    fig = result.plot()
    fig.set_size_inches(20, 10)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.set_dpi(500)
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_TimeSeries_Décomposition.png"))
    plt.close()

    # Diviser les données en ensemble d'entraînement et de test
    train = df[variable_name][:int(0.8 * len(df))]
    test = df[variable_name][int(0.8 * len(df)):]

    # Diviser les données exogènes et synchronisation des indices
    train_features = df.drop(columns=["id_Date", "id_Location", variable_name]).iloc[:int(0.8 * len(df))]
    test_features = df.drop(columns=["id_Date", "id_Location", variable_name]).iloc[int(0.8 * len(df)):]
    train_features = train_features.to_numpy()
    test_features = test_features.to_numpy()

    # Affichage ACF et PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plot_acf(train, lags=50, ax=ax1, color="green")
    plot_pacf(train, lags=50, ax=ax2, color="green")
    ax1.set_title(f"{variable_name} {station_name} - SARIMA : ACF", fontsize=15)
    ax2.set_title(f"{variable_name} {station_name} - SARIMA : PACF", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_ACF_PACF.png"))
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
    results_file = os.path.join(output_model, f"{station_name}_{variable_name}_SARIMA_Results.txt")
    with open(results_file, "w") as f:
        f.write(sarima_results.summary().as_text())

    # Prédictions du modèle SARIMA
    predictions = sarima_results.predict(start=len(train), 
                                         end=len(train) + len(test) - 1, 
                                         exog=test_features,
                                         dynamic=False)
    predictions_df = pd.DataFrame({
        "Date": predictions.index,
        f"{variable_name}_Observed": test.values,
        f"{variable_name}_Predicted": predictions.values
    })

    # Enregistrement des prédictions du modèle SARIMA
    output_predictions = base_dir / "times_series_modeling_results"
    predictions_file = os.path.join(output_predictions, f"{station_name}_SARIMA_Predictions.csv")
    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)
        existing_df["Date"] = pd.to_datetime(existing_df["Date"])
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        merged_df = pd.merge(existing_df, predictions_df, on="Date", how="outer")
    else:
        merged_df = predictions_df
    merged_df.to_csv(predictions_file, index=False)
    print(f"\nLes prédictions SARIMA pour {variable_name} {station_name} ont été enregistrées \n")

    # Visualisation des prédictions
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train, label="Train", color="black", linewidth=0.8)  # Utiliser les dates du train
    plt.plot(test.index, test, label="Test", color="gray", linewidth=0.8)  # Utiliser les dates du test
    plt.plot(predictions.index, predictions, label="Prédictions", color="red", linestyle="dotted")  # Utiliser les dates des prédictions
    plt.title(f"{variable_name} {station_name} - SARIMA : Prédictions du modèle", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel(f"{variable_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_SARIMA_Prédictions.png"))
    plt.close()

    # Évaluation du modèle
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    evaluation_file = os.path.join(output_model, f"{station_name}_{variable_name}_Évaluation_modèle.txt")
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
    Fonction pour modéliser et prédire les séries temporelles avec Prophet.

    Args:
        df (pd.DataFrame): DataFrame contenant les données avec les colonnes ["id_Date", variable_name].
        station_name (str): Nom de la station pour les visualisations et fichiers.
        variable_name (str): Nom de la variable à modéliser, par défaut "MaxTemp".
        output_model (Path): Répertoire où sauvegarder les fichiers de sortie.

    Returns:
        dict: Résultats d'évaluation du modèle (MSE, RMSE, MAE, R2, MAPE).
    """
    # Vérifier que le répertoire de sortie existe
    os.makedirs(output_model, exist_ok=True)

    # Préparer les données pour Prophet
    prophet_df = df[["id_Date", variable_name]].rename(columns={"id_Date": "ds", variable_name: "y"})
    
    # Diviser les données en train et test
    train = prophet_df[:int(0.8 * len(prophet_df))]
    test = prophet_df[int(0.8 * len(prophet_df)):]
    
    # Création et ajustement du modèle 
    model = Prophet()
    model.fit(train)

    # Prédictions du modèle Prophet
    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"]) # S'assurer que les dates de test sont dans le même format
    test["ds"] = pd.to_datetime(test["ds"])
    predictions_df = pd.DataFrame({ # Créer un dataframe avec les observations et les prédictions
        "Date": test["ds"],
        f"{variable_name}_Observed": test["y"],
        f"{variable_name}_Predicted": forecast.loc[forecast["ds"].isin(test["ds"]), "yhat"].values
    })

    # Enregister les prédictions du modèle Prophet
    output_predictions = base_dir / "times_series_modeling_results"
    predictions_file = os.path.join(output_predictions, f"{station_name}_Prophet_Predictions.csv")
    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)
        existing_df["Date"] = pd.to_datetime(existing_df["Date"])
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        merged_df = pd.merge(existing_df, predictions_df, on="Date", how="outer")
    else:
        merged_df = predictions_df
    merged_df.to_csv(predictions_file, index=False)
    print(f"\nLes prédictions Prophet pour {variable_name} {station_name} ont été enregistrées \n")

    # Visualisation des prédictions
    fig = model.plot(forecast)
    ax = fig.gca()
    for line in ax.get_lines():
        line.set_color("firebrick")
        line.set_linewidth(0.8)
    ax.collections[0].set_facecolor("gray")
    ax.collections[0].set_alpha(0.25)
    plt.title(f"{variable_name} {station_name} - Prophet : Prédictions du modèle", fontsize=15)
    plt.xlabel("Date")
    plt.ylabel(f"{variable_name}")
    fig.set_size_inches(20, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_Prophet_Predictions.png"))
    plt.close()

    # Sauvegarder les composantes du modèle
    fig_components = model.plot_components(forecast)
    fig_components.savefig(os.path.join(output_model, f"{station_name}_{variable_name}_Prophet_Components.png"))
    plt.close()

    # Évaluation des performances
    forecast = forecast.set_index("ds")
    predictions = forecast.loc[test["ds"], "yhat"]
    mse = mean_squared_error(test["y"], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["y"], predictions)
    r2 = r2_score(test["y"], predictions)
    mape = np.mean(np.abs((test["y"] - predictions) / test["y"])) * 100

    # Sauvegarder les résultats
    evaluation_file = os.path.join(output_model, f"{station_name}_{variable_name}_Prophet_Evaluation.txt")
    with open(evaluation_file, "w") as f:
        f.write(f"Evaluation du modèle Prophet pour {variable_name} {station_name} \n\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"MAPE: {mape}%\n")

    # Retourner les métriques
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


########################################################################################################

## Sélection des Location étudiées

#station_names = ["Sydney", "Adelaide", "AliceSprings", "Brisbane", "Cairns", "Canberra", "Darwin", "Hobart", "Melbourne", "Perth", "Uluru"]
station_names = ["Sydney"]

variable_name = "MaxTemp"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "MinTemp"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Rainfall"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Evaporation"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Sunshine"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "WindGustSpeed"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Humidity9am"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Humidity3pm"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Pressure9am"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Pressure3pm"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Cloud9am"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)

variable_name = "Cloud3pm"
for station_name in station_names:
    df_location, output_model = location_selection(station_name, base_dir)
    sarima_results = model_sarima(df_location, station_name, variable_name, output_model)
    prophet_results = model_prophet(df_location, station_name, variable_name, output_model)
