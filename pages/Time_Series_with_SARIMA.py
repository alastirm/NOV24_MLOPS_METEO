import sys
import os
import emoji
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO


url = "data/weatherAUS.csv"
df = pd.read_csv(url)

st.set_page_config(page_title="MeteoStralia",
                   layout="wide",
                   page_icon = emoji.emojize('🦘'))

with st.container():

    st.subheader("Modelling & Prediction of meteorological variables with SARIMA")
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Select a station and a variable to study")
    st.markdown("<br>", unsafe_allow_html=True)

    location, variable = st.columns(2)

    with location:
        city = st.selectbox(label="Select a Station",
                            options=sorted(df["Location"].unique()),
                            index=None,
                            label_visibility="visible",
                            help="Select a station",
                            placeholder="No station selected")
    with variable:
        var_to_study = st.selectbox(label="Select a Variable",
                                    options=df.columns[2:],
                                    index=None,
                                    help="Select a variable to analyze",
                                    placeholder="No variable selected")
    st.markdown("<br>", unsafe_allow_html=True)


if city and var_to_study:

    # Charger les données spécifiques à la ville depuis un fichier CSV
    base_dir = "src/data_location_V2"
    path_dir = os.path.join(base_dir, f"df_{city}.csv")
    city_df = pd.read_csv(path_dir)
    city_df["id_Date"] = pd.to_datetime(city_df["id_Date"])
    city_df = city_df.sort_values(by="id_Date", ascending=True)
    city_df = city_df.loc[:, ~city_df.columns.str.contains("Year", case=False)]
    city_data = df[df["Location"] == city].copy()
    city_data.dropna(subset=[var_to_study], inplace=True)
    city_data["Date"] = pd.to_datetime(city_data["Date"])
    city_data.set_index("Date", inplace=True)
    ts = city_data[var_to_study]

    # Décomposition de la série temporelle
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Time series decomposition")
    st.markdown("""
        <style>
        .custom-box {
            background-color: #c1c6fc;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-family: monospace;
        }
        </style>

        <div class="custom-box">
        Analysis of time series begins with a decomposition of the serie :
        </div>
    """, unsafe_allow_html=True)
    result = seasonal_decompose(city_data[var_to_study], model="additive", period=365)
    trace_original = go.Scatter(x=result.observed.index, y=result.observed, mode="lines", name="Original", line=dict(color="gray", width=1))
    trace_trend = go.Scatter(x=result.trend.index, y=result.trend, mode="lines", name="Trend", line=dict(color="darkolivegreen", width=1))
    trace_seasonal = go.Scatter(x=result.seasonal.index, y=result.seasonal, mode="lines", name="Seasonal", line=dict(color="olivedrab", width=1))
    trace_resid = go.Scatter(x=result.resid.index, y=result.resid, mode="markers", name="Residual", marker=dict(color="yellowgreen", size=5))
    fig = sp.make_subplots(
        rows=4, cols=1,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
        shared_xaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    fig.add_trace(trace_original, row=1, col=1)
    fig.add_trace(trace_trend, row=2, col=1)
    fig.add_trace(trace_seasonal, row=3, col=1)
    fig.add_trace(trace_resid, row=4, col=1)
    fig.update_layout(
        height=1200,
        showlegend=True
    )
    st.plotly_chart(fig)

    # Diviser les données en ensemble d'entraînement et de test
    train = city_df[var_to_study][:int(0.8 * len(city_df))]
    test = city_df[var_to_study][int(0.8 * len(city_df)):]

    # Séparation des caractéristiques (features) dans `city_df` (pas `city_data` ici)
    train_features = city_df.drop(columns=["id_Date", "id_Location", var_to_study]).iloc[:int(0.8 * len(city_df))]
    test_features = city_df.drop(columns=["id_Date", "id_Location", var_to_study]).iloc[int(0.8 * len(city_df)):]

    # Visualisation de l'ACF et PACF
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Autocorrelation and Partial Autocorrelation")
    st.markdown("""
        <style>
        .custom-box {
            background-color: #c1c6fc;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-family: monospace;
        }
        </style>

        <div class="custom-box">
        Autocorrelation and partial autocorrelation graphs help to understand the properties of time series
        and to determine the appropriate parameters for SARIMA Model :
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    lags = 50
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="none")
    plot_acf(train, lags=lags, ax=ax1, color="dodgerblue")
    plot_pacf(train, lags=lags, ax=ax2, color="dodgerblue")
    ax1.set_title("ACF", fontsize=12, color="black")
    ax2.set_title("PACF", fontsize=12, color="black")
    ax1.set_facecolor("none")
    ax2.set_facecolor("none")
    for ax in [ax1, ax2]:
        ax.spines["bottom"].set_color("black")
        ax.spines["top"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
    fig.patch.set_alpha(0)
    st.pyplot(fig)
    st.markdown("<br>", unsafe_allow_html=True)

    # Création et ajustement du modèle SARIMA
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("SARIMA Model")
    st.markdown("""
        <style>
        .custom-box {
            background-color: #c1c6fc;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-family: monospace;
        }
        </style>

        <div class="custom-box">
        Statistical model used to analyze and forecast long time series, including seasonal variations :
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.code("""
            sarima_model = SARIMAX(train,
                                   exog = train_features,              # Include exogenous variables that may influence the target variable
                                   order = (1, 1, 1),
                                   seasonal_order = (1, 1, 1, 12),     # 12 for annual seasonality
                                   enforce_stationarity = False,       # Accept non-stationary series
                                   enforce_invertibility = False)
            """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    sarima_model = SARIMAX(train,
                           exog=train_features,
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_results = sarima_model.fit(method="powell", disp=False)
    st.write(sarima_results.summary())

    # Prédictions du modèle SARIMA
    predictions = sarima_results.predict(start=len(train),
                                         end=len(train) + len(test) - 1,
                                         exog=test_features,
                                         dynamic=False)

    # Visualisation des prédictions
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model predictions")
    st.markdown("""
        <style>
        .custom-box {
            background-color: #c1c6fc;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-family: monospace;
        }
        </style>

        <div class="custom-box">
        Visual representation of model predictions :
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(15, 5), facecolor="none")
    ax.plot(train.index, train, label="Train", color="black", linewidth=0.7)
    ax.plot(test.index, test, label="Test", color="gray", linewidth=0.7)
    ax.plot(test.index, predictions, label="Predictions", color="red", linestyle="dotted", linewidth=0.7)
    ax.set_xlabel("Date", color="black")
    ax.set_ylabel(f"{var_to_study}", color="black")
    ax.legend(labelcolor="black", frameon=False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")
    st.pyplot(fig)

    # Évaluation du modèle
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model evaluation")
    st.markdown("""
        <style>
        .custom-box {
            background-color: #c1c6fc;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-family: monospace;
        }
        </style>

        <div class="custom-box">
        Several metrics are used to assess model performance :
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    r2 = round(r2_score(test, predictions), 3)
    mse = round(mean_squared_error(test, predictions), 3)
    rmse = round(np.sqrt(mse), 3)
    mae = round(mean_absolute_error(test, predictions), 3)
    st.write(f"📈 R2 : {r2}")
    st.write(f"📊 MSE : {mse}")
    st.write(f"📊 RMSE : {rmse}")
    st.write(f"📉 MAE : {mae}")
