# Importer les bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
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

# Configurer les options pandas et l'encodage de la sortie
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("future.no_silent_downcasting", True)
sys.stdout.reconfigure(encoding="utf-8")
warnings.simplefilter("ignore", ConvergenceWarning)


########################################################################################################

## Lecture du dataframe & Sélection des données

# Chargement des données
df_final = pd.read_csv("C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\VSCode Env\\df_final.csv")
df_final = df_final.rename(columns={"id_Location": "Location", "id_Date": "Date"})
df_final["Date"] = pd.to_datetime(df_final["Date"])

# Ré-organisation du dataframe
df_final = df_final.drop(columns=["Climate_Desert", "Climate_Grassland", "Climate_Subtropical", "Climate_Temperate", "Climate_Tropical",])
columns_order = ["Location", "Date", "Year", "Month", "Season_Autumn", "Season_Spring", "Season_Summer", "Season_Winter",
                "MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", 
                "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindGustDir_cos", "WindGustDir_sin", "WindDir9am_cos", "WindDir9am_sin", "WindDir3pm_cos", "WindDir3pm_sin",
                "Rainfall", "RainToday", "RainTomorrow"]
df_final = df_final[columns_order]
df_final = df_final.sort_values(by="Date", ascending=True)

# Grouper les données par Location
output_folder = "C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\Data\\df_location"
groupes = df_final.groupby("Location")
for location, groupe in groupes:
    nouveau_df = groupe.copy()
    output_path = os.path.join(output_folder, f"df_{location}.csv")
    nouveau_df.to_csv(output_path, index=False)

### J'ai essayer de faire sans créer de fichiers csv intermédiaires, mais le reste du code si je ne fait pas ça !!!!


########################################################################################################

## Sélection de la Sation/Ville étudiée

### Attention à bien changer le nom de la ville ici ###

# Location étudiée
station_name = "Perth"
df_location = pd.read_csv("C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\Data\\df_location\\df_Perth.csv")
df_location["Date"] = pd.to_datetime(df_location["Date"])

# Infos sur le dataframe
print("Infos sur df_location :")
print(df_location.info(), "\n")
print("Head df_location :", df_location.head())

# Créer un dossier pour sauvegarder les fichiers
output_dir = "C:\\Users\\JenniferLaurent\\Desktop\\Data Scientist\\Projet\\Modelling\\Perth"
os.makedirs(output_dir, exist_ok=True)


########################################################################################################

## Modèle SARIMA

# Visualisation de la série temporelle de MinTemp
plt.figure(figsize=(15, 7))
plt.plot(df_location["Date"], df_location["MinTemp"], linewidth=0.8, label="Température minimale", color="mediumblue") 
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel(f"Température minimale (°C)")
plt.title(f"MinTemp {station_name} - SARIMA : Évolution de la variable", fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"1. {station_name}_MinTemp_Évolution.png"))
plt.close() 

# Décomposition de la série temporelle de MinTemp
plt.figure(figsize=(15, 7))
plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", ["royalblue", "royalblue", "royalblue", "royalblue"])})
result = seasonal_decompose(df_location["MinTemp"], model="additive", period=365)
fig = result.plot()
fig.axes[0].plot(result.observed, color="mediumblue", linewidth=0.5)
fig.axes[1].plot(result.trend, color="mediumblue", linewidth=0.5)
fig.axes[2].plot(result.seasonal, color="mediumblue", linewidth=0.5)
fig.axes[3].scatter(result.resid.index, result.resid, color="mediumblue")
for ax in fig.axes:
    ax.set_title("") # Enlever les titres des sous-graphiques
fig.suptitle(f"MinTemp {station_name} - SARIMA : Décomposition de la série temporelle", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"2. {station_name}_MinTemp_TimeSeries_Décomposition.png"))
plt.show()

# Diviser les données en ensemble d'entraînement et de test
train = df_location["MinTemp"][:int(0.8 * len(df_location))]
test = df_location["MinTemp"][int(0.8 * len(df_location)):]

# Créer un DataFrame des features
train_features = df_location.drop(columns=["Date", "Location", "MinTemp"])[:int(0.8 * len(df_location))]
test_features = df_location.drop(columns=["Date", "Location", "MinTemp"])[int(0.8 * len(df_location)):]

# Afficher ACF et PACF de MinTemp
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
plot_acf(train, lags=50, ax=ax1, color="mediumblue")
ax1.set_title(f"MinTemp {station_name} - SARIMA : ACF", fontsize=15)
plot_pacf(train, lags=50, ax=ax2, color="mediumblue")
ax2.set_title(f"MinTemp {station_name} - SARIMA : PACF", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"3. {station_name}_MinTemp_ACF_PACF.png"))
plt.close() 

# Créer et ajuster le modèle SARIMA avec les régressseurs externes
sarima_model = SARIMAX(train, 
                       exog=train_features,  # Utilisation des régressseurs externes pour l'entraînement
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12), 
                       enforce_stationarity=False, 
                       enforce_invertibility=False)
sarima_results = sarima_model.fit()
print(sarima_results.summary())

# Sauvegarder les résultats détaillés du modèle SARIMA
results_file = os.path.join(output_dir, f"4. {station_name}_MinTemp_SARIMA_Results.txt")
with open(results_file, "w") as f:
    f.write(sarima_results.summary().as_text())

# Prédictions avec les régressseurs externes
predictions = sarima_results.predict(start=test.index[0], 
                                     end=test.index[-1], 
                                     exog=test_features,  # Utilisation des régressseurs pour les prédictions
                                     dynamic=False)

# Visualiser les prédictions du modèle SARIMA pour MinTemp
plt.figure(figsize=(15, 7))
plt.plot(train, label="Train", color="black", linewidth=0.8)
plt.plot(test, label="Test", color="gray", linewidth=0.8)
plt.plot(predictions, label="Prédictions", color="royalblue", linestyle="dotted")
plt.title(f"MinTemp {station_name} - SARIMA : Prédictions du modèle", fontsize=15)
plt.xlabel("Date")
plt.ylabel("MinTemp")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"5. {station_name}_MinTemp_SARIMA_Prédictions.png"))
plt.close() 

# Évaluation du modèle SARIMA pour MinTemp
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, predictions)
r2 = r2_score(test, predictions)
mape = np.mean(np.abs((test - predictions) / test)) * 100

# Sauvegarder l'évaluation dans un fichier texte
evaluation_file = os.path.join(output_dir, f"6. {station_name}_MinTemp_Évaluation_modèle.txt")
with open(evaluation_file, "w") as f:
    f.write(f"Evaluation du modèle SARIMA pour MinTemp {station_name} \n\n")
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"Root Mean Squared Error: {rmse}\n")
    f.write(f"Mean Absolute Error: {mae}\n")
    f.write(f"R-squared: {r2}\n")
    f.write(f"Mean Absolute Percentage Error: {mape}%\n")



########################################################################################################

## Modèle Prophet

# Charger et préparer les données pour Prophet
prophet_df = df_location[["Date", "MinTemp"]].rename(columns={"Date": "ds", "MinTemp": "y"})

# Créer et configurer le modèle Prophet
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                        daily_seasonality=False, seasonality_mode="additive")

# Ajuster le modèle sur les données d'entraînement
train_prophet = prophet_df[:int(0.8 * len(prophet_df))]  # 80% pour l'entraînement
test_prophet = prophet_df[int(0.8 * len(prophet_df)):]   # 20% pour le test
prophet_model.fit(train_prophet)

# Générer des prédictions sur la période de test
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq="D")
forecast = prophet_model.predict(future)

# Visualisation des composants des prédictions
plt.figure(figsize=(15, 15))
fig_components = prophet_model.plot_components(forecast)
axes = fig_components.get_axes()
for ax in axes:
    ax.set_title(f"MinTemp {station_name} - Prophet : Composants des prédictions", fontsize=13)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
for ax in axes:
    for line in ax.get_lines():
        line.set_color("mediumblue")
    for collection in ax.collections:
        collection.set_facecolor("royalblue")
        collection.set_alpha(0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"7. {station_name}_MinTemp_Prophet_Components.png"))
plt.close()

# Visualisation des prédictions
plt.figure(figsize=(15, 6))
fig = prophet_model.plot(forecast)
ax = fig.gca()
for line in ax.get_lines():
    line.set_color("mediumblue")
ax.collections[0].set_facecolor("royalblue")
ax.collections[0].set_alpha(0.3)
plt.title(f"MinTemp {station_name} - Prophet : Prédictions", fontsize=15, pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Température maximale (°C)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"8. {station_name}_MinTemp_Prophet_Predictions.png"))
plt.close()

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(15, 7))
plt.plot(test_prophet["ds"], test_prophet["y"], label="Valeurs réelles", color="mediumblue", linewidth=1)
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
plt.savefig(os.path.join(output_dir, f"9. {station_name}_MinTemp_Prophet_Predictions_vs_Real.png"))
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
evaluation_file_prophet = os.path.join(output_dir, f"10. {station_name}_MinTemp_Prophet_Évaluation_modèle.txt")
with open(evaluation_file_prophet, "w") as f:
    f.write(f"Evaluation du modèle Prophet pour MinTemp {station_name} \n\n")
    f.write(f"Mean Squared Error: {mse_prophet}\n")
    f.write(f"Root Mean Squared Error: {rmse_prophet}\n")
    f.write(f"Mean Absolute Error: {mae_prophet}\n")
    f.write(f"R-squared: {r2_prophet}\n")
    f.write(f"Mean Absolute Percentage Error: {mape_prophet}%\n")

