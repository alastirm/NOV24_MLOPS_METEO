import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, brier_score_loss, precision_recall_curve, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

import emoji

sys.path.insert(0, '/home/mathieu/code/MathieuAmacher/datascientest/NOV24-BDS-METEO/src2')
# import preprocess
from src2 import preprocessing, modelisation

url = 'data/weatherAUS.csv'
df = pd.read_csv(url)

st.set_page_config(page_title = 'MeteoStralia',
                   layout = 'wide',
                   page_icon = emoji.emojize('🦘'))

st.html("""
        <h1 style = "color : rgb(16, 0, 97); font-size : 200%;">Location Prediction </h1>
        <h4 style = 'color : rgb(156, 4, 4);'>5 days rain prediction </h4>
    </div>
""")

with st.container(border = False):

    location, time = st.columns(2, border = True)

    with location:

        city = st.selectbox(label = 'Select a City',
                        options = sorted(df['Location'].unique()),
                        index = None,
                        label_visibility="visible",
                        help = 'get a listed city',
                        placeholder = 'No city selected yet'
                        )

    with time:

        date = st.date_input(label = 'The date of today',
                             help = 'year/month/day',
                             value = 'today')

if city :

    df_city = df[df['Location'] == city]

    with st.container(border = False):

        wind1, wind2 = st.columns(2, border = True)

        with wind1:
            st.write("How about Wind Speed ?")

            speed1, speed2, speed3 = st.columns(3)

            with speed1:

                if df_city.iloc[-2,:]['WindSpeed9am'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['WindSpeed9am'].astype(float)
                else:
                    value = 10.0

                windspeed9am = st.number_input(label = 'at 9:00 am',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = value,
                                help = 'Look at your anemometer at 9 in the morning',
                                )

            with speed2:

                if df_city.iloc[-2,:]['WindSpeed3pm'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['WindSpeed3pm'].astype(float)
                else:
                    value = 10.0

                windspeed3pm = st.number_input(label = 'at 3:00 pm',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = value,
                                help = 'Look at your anemometer at 3 in the afternoon',
                                )

            with speed3:

                if df_city.iloc[-2,:]['WindGustSpeed'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['WindGustSpeed'].astype(float)
                else:
                    value = 10.0

                windgustspeed = st.number_input(label = 'Gust',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = np.round(value, 2),
                                help = 'Look at your anemometer for the fastest wind speed of the day',
                                )

        with wind2:
            st.write("How about Wind Direction ?")

            dir1, dir2, dir3 = st.columns(3)

            with dir1:

                options = sorted(df['WindDir9am'].dropna().unique())
                default_value = df_city.iloc[-2,:]['WindDir9am'] if df_city.iloc[-2,:]['WindDir9am'] != ' ' else df_city['WindDir9am'].mode()[0]

                if default_value not in options:
                    default_value = options[0]

                winddir9am = st.selectbox(label = '9:00 am',
                                        options = sorted(df['WindDir9am'].dropna().unique()),
                                        help = 'Look at your compass at 9:00 am',
                                        index = options.index(default_value)
                                        )

            with dir2:

                options = sorted(df['WindDir3pm'].dropna().unique())
                default_value = df_city.iloc[-2,:]['WindDir3pm'] if df_city.iloc[-2,:]['WindDir3pm'] != ' ' else df_city['WindDir3pm'].mode()[0]

                if default_value not in options:
                    default_value = options[0]

                winddir3pm = st.selectbox(label = '3:00 pm',
                                        options = sorted(df['WindDir3pm'].dropna().unique()),
                                        help = 'Look at your compass at 3:00 pm',
                                        index = options.index(default_value),
                                        )

            with dir3:

                options = sorted(df['WindGustDir'].dropna().unique())
                default_value = df_city.iloc[-2,:]['WindGustDir'] if df_city.iloc[-2,:]['WindGustDir'] != ' ' else df_city['WindGustDir'].mode()[0]

                if default_value not in options:
                    default_value = options[0]

                windgustdir = st.selectbox(label = 'Gust',
                                        options = sorted(df['WindGustDir'].dropna().unique()),
                                        help = 'Look at your when the gust blow',
                                        index = options.index(default_value),
                                        )


    with st.container(border = False):

        temp_mm, temp_h = st.columns(2, border = True)

        with temp_mm:

            if df_city.iloc[-2,:]['MinTemp'].astype(float) != 0:
                    value_min = df_city.iloc[-2,:]['MinTemp'].astype(float)
            else:
                value_min = df_city['MinTemp'].mode()[0].astype(float)

            if df_city.iloc[-2,:]['MaxTemp'].astype(float) != 0:
                    value_max = df_city.iloc[-2,:]['MaxTemp'].astype(float)
            else:
                value_max = df_city['MaxTemp'].mode()[0].astype(float)

            st.write("Select the min and max temperature")

            MinTemp, MaxTemp = st.select_slider(
                label = 'slide it',
                options = np.round(np.arange(-20, 50, 0.1), 2),
                value=(value_min, value_max),
            )

        with temp_h:

            st.write('What is the temperature at ?')

            temp9, temp3 = st.columns(2, border = False)

            with temp9:

                if df_city.iloc[-2,:]['Temp9am'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['Temp9am'].astype(float)
                else:
                    value = df_city['Temp9am'].mode()[0].astype(float)

                Temp9am = st.number_input(label = '9:00 am',
                                min_value = -20.0,
                                max_value = 50.0,
                                value = value,
                                help = 'Look at your thermometer at 9 in the morning',
                                )

            with temp3:

                if df_city.iloc[-2,:]['Temp3pm'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['Temp3pm'].astype(float)
                else:
                    value = df_city['Temp9am'].mode()[0].astype(float)

                Temp3pm =  st.number_input(label = '3:00 pm',
                                min_value = -20.0,
                                max_value = 50.0,
                                value = value,
                                help = 'Look at your thermometer at 3 in the afternoon',
                                )

    with st.container(border = False):

        press, cloud = st.columns(2, border = True)

        with press:

            st.write('Pressure at 9:00 am and 3:00 pm')

            press9, press3 = st.columns(2, border = False)

            with press9:

                if df_city.iloc[-2,:]['Pressure9am'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['Pressure9am'].astype(float)
                else:
                    value = df_city['Pressure9am'].mode()[0].astype(float)

                Pressure9am = st.number_input(label = 'at 9:00 am',
                                    min_value = 970.0,
                                    max_value = 1050.0,
                                    value = value,
                                    help = 'Look at your barometer at 9 in the morning',
                                    )

            with press3:

                if df_city.iloc[-2,:]['Pressure3pm'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['Pressure3pm'].astype(float)
                else:
                    value = df_city['Pressure3pm'].mode()[0].astype(float)

                Pressure3pm = st.number_input(label = 'at 3:00 pm',
                                    min_value = 970.0,
                                    max_value = 1050.0,
                                    value = value,
                                    help = 'Look at your barometer at 3 in the afternoon',
                                    )

        with cloud:

            st.write('Give the nebulosity ?')

            cloud9, cloud3 = st.columns(2, border = False)

            with cloud9:

                options = sorted(df['Cloud9am'].dropna().unique())
                default_value = df_city.iloc[-2,:]['Cloud9am'] if df_city.iloc[-2,:]['Cloud9am'] != ' ' else df_city['Cloud9am'].mode()[0]

                if default_value not in options:
                    default_value = options[0]

                Cloud9am = st.selectbox(label = '9:00 am',
                                options = np.arange(0, 10, 1),
                                help = 'Look at the sky',
                                index = options.index(default_value),
                                )

            with cloud3:

                options = sorted(df['Cloud3pm'].dropna().unique())
                default_value = df_city.iloc[-2,:]['Cloud3pm'] if df_city.iloc[-2,:]['Cloud3pm'] != ' ' else df_city['Cloud3pm'].mode()[0]

                if default_value not in options:
                    default_value = options[0]

                Cloud3pm = st.selectbox(label = '3:00 pm',
                                options = np.arange(0, 10, 1),
                                help = 'Look at the sky',
                                index = options.index(default_value),
                                )

    with st.container(border = False):

        hum, evap= st.columns(2, border = True)

        with hum:

            st.write('What is the humidity ?')

            hum9, hum3 = st.columns(2, border = False)

            with hum9:

                # if df_city.iloc[-2,:]['Humidity9am'].astype(float) != 0:
                #     value = df_city.iloc[-2,:]['Humidity9am'].astype(float)
                # else:
                value = df_city['Humidity9am'].median().astype(float)

                Humidity9am = st.slider(
                    label = '9:00 am',
                    min_value = 0.0,
                    max_value = 100.0,
                    value = value,
                    step = 1.0,
                    help = 'give the humidity at 9:00 am'
                )

            with hum3:

                if df_city.iloc[-2,:]['Humidity3pm'].astype(float) != 0:
                    value = df_city.iloc[-2,:]['Humidity3pm'].astype(float)
                else:
                    value = df_city['Humidity3pm'].mode()[0].astype(float)


                Humidity3pm = st.slider(
                    label = '3:00 pm',
                    min_value = 0.0,
                    max_value = 100.0,
                    value = value,
                    step = 1.0,
                    help = 'give the humidity at 3:00 pm'
                )

        with evap:

            st.write('Do you know the amount of evaporation ?')

            if pd.isna(df_city.iloc[-2,:]['Evaporation']):
                value = 0.0
            else:
                value = float(df_city.iloc[-2,:]['Evaporation'])


            Evaporation = st.slider(
                label = 'in mm',
                min_value = 0.0,
                max_value = 21.0,
                value = value,
                step = 1.0,
                help = 'hard help yourself',
            )



    with st.container(border = True):

        sunsh, raintoday = st.columns(2, border = False)

        with sunsh:

            if pd.isna(df_city.iloc[-2,:]['Sunshine']):
                value = 0.0
            else:
                value = float(df_city.iloc[-2,:]['Evaporation'])


            Sunshine = st.number_input(label = 'How many hours of sun today ?',
                                        min_value = 0.0,
                                        max_value = 14.5,
                                        step = 0.25,
                                        value = value,
                                        help = 'that is obsvious',
                                    )

        with raintoday :

            if df_city.iloc[-2,:]['RainToday'] == 'yes':
                    value = True
            else:
                value = False

            RainToday = st.checkbox(label = 'Does it rain today ?',
                                    value = value,
            )

            if RainToday:
                RainToday_value = 'Yes'
            else:
                RainToday_value = 'No'

            if RainToday:

                Rainfall = st.number_input(label = 'How many minimeter of rain roday? ',
                                        min_value = 0.0,
                                        max_value = 371.0,
                                        value = np.round(df_city['Rainfall'].mean(), 2),
                                        help = 'that is obsvious',
                                    )
            else :
                Rainfall = 0

    if st.button("Rain Tomorrow ?", type="primary"):


        df_day = pd.DataFrame(
            {'Date' : [pd.to_datetime(date).strftime('%Y-%m-%d')],
            'MinTemp' : [MinTemp],
            'MaxTemp' :  [MaxTemp],
            'Rainfall' : [Rainfall],
            'Evaporation' : [Evaporation],
            'Sunshine' : [Sunshine],
            'WindGustDir' : [windgustdir],
            'WindGustSpeed' : [np.round(windgustspeed, 2)],
            'Temp9am' : [Temp9am],
            'Humidity9am' : [Humidity9am],
            'Cloud9am' : [Cloud9am],
            'WindDir9am' : [winddir9am],
            'WindSpeed9am' : [windspeed9am],
            'Pressure9am' : [Pressure9am],
            'Temp3pm': [Temp3pm],
            'Humidity3pm' : [Humidity3pm],
            'Cloud3pm' : [Cloud3pm],
            'WindDir3pm' : [winddir3pm],
            'WindSpeed3pm' : [windspeed3pm],
            'Pressure3pm' : [Pressure3pm],
            'Location' : [city],
            'RainToday' : [str(RainToday_value)],
            'RainTomorrow' : str('Yes')
        })


#################################################
# df complet

        df_full = pd.concat([df_city, df_day], axis = 0)

###########################################
# prévision sur 5 jour

        df_full['Date'] = pd.to_datetime(df_full['Date'])
        last_date = df_full['Date'].max()
        start_date = last_date + pd.Timedelta(days=1)
        end_date = last_date + pd.Timedelta(days=6)
        days_to_add = pd.date_range(start=start_date, end=end_date, freq='D')

        prev = []

        for day in days_to_add:
            day_data = df_full[(df_full['Date'].dt.month == day.month) & (df_full['Date'].dt.day == day.day)]
            numeric_columns = day_data.select_dtypes(include=['number']).columns
            object_columns = day_data.select_dtypes(include=['object']).columns
            mean_values = day_data[numeric_columns].mean()
            mode_values = day_data[object_columns].mode().iloc[0]
            combined_values = pd.concat([mean_values, mode_values])
            combined_values['Date'] = day
            prev.append(combined_values)

        prev = pd.DataFrame(prev)

############################################
#
        df_full2 = pd.concat([df_full, prev], axis = 0)



        df_preproc = preprocessing(df_full2).dropna()
        cible = pd.DataFrame(df_preproc.iloc[-6 :, :]).drop(columns = 'RainTomorrow')

        X = df_preproc.drop(columns = 'RainTomorrow')
        y = df_preproc['RainTomorrow']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

        scaler = MinMaxScaler(feature_range = (-1, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        cible_scaled = scaler.transform(cible)

        #  y_train.value_counts(normalize = True)


        model = LogisticRegression(verbose = 0,
                                C = 1,
                                max_iter = 1000,
                                penalty = 'l1',
                                solver = 'liblinear',
                                intercept_scaling = 0.1,
                                l1_ratio = 0.5,
                                tol = 0.01)

        model.fit(X_train_scaled, y_train)

        y_pred_prob1 = model.predict_proba(X_test_scaled)[:, 1]

        # if y_train.value_counts(normalize = True)[1] < 0.15:
        #         threshold = 0.165
        # if y_train.value_counts(normalize = True)[1] > 0.30:
        #     threshold = 0.85
        # else:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob1)
        diff = abs(precision - recall)
        threshold = thresholds[diff.argmin()]

        y_pred = (y_pred_prob1 > threshold).astype(int)

        proba = model.predict_proba(cible_scaled)



        with st.container(border = False):

            predict, seuil = st.columns(2, border = True)

            with predict:

                image, result, confidence = st.columns(3, border = False)

                proba0 = proba[0]

                with result:

                    st.write('Sky like :')

                    if proba[0][0] == 0.0 :
                        st.write('No Rain Tomorrow')
                    else:
                        st.write('Rain Tomorrow')

                with image:

                    if proba0[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                    elif proba0[1] >= threshold:
                        st.image('image/niv4.png')
                        st.write('rain')
                    elif proba0[1] > threshold/2:
                        st.image('image/niv3.png')
                        st.write('cloudy')
                    elif proba0[1] > threshold/3:
                        st.image('image/niv2.png')
                        st.write('little cloudy')
                    else :
                        st.image('image/niv1.png')
                        st.write('plain sun')

                with confidence:

                    st.write('Date :')
                    st.write(cible.index[0].date())

                    acc = accuracy_score(y_test, y_pred)
                    st.write('Confidence Rate :', round((acc), 2)*100, '%')


            with seuil:

                fig, ax = plt.subplots(figsize=(8, 1))
                ax.plot([0, proba[0][0]], [0.2, 0.2], color = 'blue', linewidth = 20)
                ax.plot([proba[0][0], 1], [0.2, 0.2], color = 'red', linewidth = 20)
                ax.plot([1 - threshold, 1 - threshold],[0.15, 0.25], color = 'orange', linewidth = 5)

                ax.set_yticks([])
                st.pyplot(fig)

                st.write('Probability No Rain :', round(proba[0][0]*100, 2), '%')
                st.write('Probability Rain :', round(proba[0][1]*100, 2), '%')
                st.write('Threshold', round((1-threshold)*100, 2), '%')

        with st.container(border = False):

            prob1, prob2, prob3, prob4, prob5 = st.columns(5, border = True)

            with prob1:

                proba1 = proba[1]

                st.write(cible.index[1].date())

                if proba1[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                elif proba1[1] >= threshold:
                    st.image('image/niv4.png')
                    st.write('rain')
                elif proba1[1] > threshold/2:
                    st.image('image/niv3.png')
                    st.write('cloudy')
                elif proba1[1] > threshold/3:
                    st.image('image/niv2.png')
                    st.write('little cloudy')
                else :
                    st.image('image/niv1.png')
                    st.write('plain sun')

            with prob2:

                proba2 = proba[2]

                st.write(cible.index[2].date())

                if proba2[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                elif proba2[1] >= threshold:
                    st.image('image/niv4.png')
                    st.write('rain')
                elif proba2[1] > threshold/2:
                    st.image('image/niv3.png')
                    st.write('cloudy')
                elif proba2[1] > threshold/3:
                    st.image('image/niv2.png')
                    st.write('little cloudy')
                else :
                    st.image('image/niv1.png')
                    st.write('plain sun')

            with prob3:

                proba3 = proba[3]

                st.write(cible.index[3].date())

                if proba3[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                elif proba3[1] >= threshold:
                    st.image('image/niv4.png')
                    st.write('rain')
                elif proba3[1] > threshold/2:
                    st.image('image/niv3.png')
                    st.write('cloudy')
                elif proba3[1] > threshold/3:
                    st.image('image/niv2.png')
                    st.write('little cloudy')
                else :
                    st.image('image/niv1.png')
                    st.write('plain sun')

            with prob4:

                proba4 = proba[4]

                st.write(cible.index[4].date())

                if proba4[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                elif proba4[1] >= threshold:
                    st.image('image/niv4.png')
                    st.write('rain')
                elif proba4[1] > threshold/2:
                    st.image('image/niv3.png')
                    st.write('cloudy')
                elif proba4[1] > threshold/3:
                    st.image('image/niv2.png')
                    st.write('little cloudy')
                else :
                    st.image('image/niv1.png')
                    st.write('plain sun')

            with prob5:

                proba5 = proba[5]

                st.write(cible.index[5].date())

                if proba5[1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                elif proba5[1] >= threshold:
                    st.image('image/niv4.png')
                    st.write('rain')
                elif proba5[1] > threshold/2:
                    st.image('image/niv3.png')
                    st.write('cloudy')
                elif proba5[1] > threshold/3:
                    st.image('image/niv2.png')
                    st.write('little cloudy')
                else :
                    st.image('image/niv1.png')
                    st.write('plain sun')



        st.write('ok')
