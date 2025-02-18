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
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO


url = "data/weatherAUS.csv"
df = pd.read_csv(url)

st.set_page_config(page_title="MeteoStralia",
                   layout="wide",
                   page_icon=emoji.emojize(":thumbs_up:"))

with st.container():

    st.subheader("Modelling & Prediction of meteorological variables with Prophet")
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

    # Préparer les données pour Prophet
    prophet_df = city_df[["id_Date", var_to_study]].rename(columns={"id_Date": "ds", var_to_study: "y"})

    # Diviser les données en train et test
    train = prophet_df[:int(0.8 * len(prophet_df))]
    test = prophet_df[int(0.8 * len(prophet_df)):].copy()
    test["ds"] = pd.to_datetime(test["ds"])

    # Création et ajustement du modèle Prophet
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Prophet Model")
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
        Open-source library developed by Meta, specially designed for forecasting time series with non-linear trends
        and pronounced seasonal effects :
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.code(""" 
            model = Prophet()
            model.fit(train)
            """, language="python")
    st.markdown("<br>", unsafe_allow_html=True)
    model = Prophet()
    model.fit(train)

    # Prédictions du modèle Prophet
    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast_test = forecast[forecast["ds"].isin(test["ds"])]  #
    if len(forecast_test) != len(test):
        print(f"Attention : longueur différente entre test ({len(test)}) et forecast_test ({len(forecast_test)})")
        test = test[test["ds"].isin(forecast_test["ds"])]
    predictions = forecast_test["yhat"].values    

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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["ds"], 
        y=forecast["yhat"], 
        mode="lines", 
        name="Predictions", 
        line=dict(color="red", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], 
        y=forecast["yhat_upper"], 
        mode="lines", 
        name="Upper interval", 
        line=dict(color="lightgray", dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], 
        y=forecast["yhat_lower"], 
        mode="lines", 
        name="Lower interval", 
        line=dict(color="lightgray", dash="dot"),
        fill="tonexty",
        fillcolor="rgba(169,169,169,0.2)"
    ))
    fig.add_trace(go.Scatter(
        x=test["ds"], 
        y=test["y"], 
        mode="markers", 
        name="Real data", 
        marker=dict(color="black", size=4)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=var_to_study,
        template="plotly_white",
        showlegend=True,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

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
    r2 = round(r2_score(test["y"], predictions), 3)
    mse = round(mean_squared_error(test["y"], predictions), 3)
    rmse = round(np.sqrt(mse), 3)
    mae = round(mean_absolute_error(test["y"], predictions), 3)
    st.write(f"📈 R2 : {r2}")
    st.write(f"📊 MSE : {mse}")
    st.write(f"📊 RMSE : {rmse}")
    st.write(f"📉 MAE : {mae}")