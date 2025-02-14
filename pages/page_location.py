import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

st.write('Here you can fin tune forecast')

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

        df_full = pd.concat([df_city, df_day], axis = 0)

        df_preproc = preprocessing(df_full).dropna()
        cible = pd.DataFrame(df_preproc.iloc[-1, :]).T.drop(columns = 'RainTomorrow')

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

        if y_train.value_counts(normalize = True)[1] < 0.15:
                threshold = 0.165
        if y_train.value_counts(normalize = True)[1] > 0.30:
            threshold = 0.85
        else:
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob1)
            diff = abs(precision - recall)
            threshold = thresholds[diff.argmin()]

        y_pred = (y_pred_prob1 > threshold).astype(int)


        # score_accuracy = model.score(X_test_scaled, y_test)

        # print('score accuracy : ', score_accuracy)
        # print('f1 score : ', f1_score(y_test, y_pred))
        # print('roc-auc score : ', roc_auc_score(y_test, y_pred))
        # print('brier score : ', brier_score_loss(y_test, y_pred), '\n\n')

        # st.write(confusion_matrix(y_test, y_pred), '\n\n')

        with st.container(border = False):

            predict, seuil = st.columns(2, border = True)

            with predict:

                image, result, confidence = st.columns(3, border = True)

                with result:
                    prediction = model.predict(cible_scaled)

                    st.write('Sky like :')

                    if prediction[0] == 0.0 :
                        st.write('No Rain Tomorrow')
                    else:
                        st.write('Rain Tomorrow')

                with image:

                    proba = model.predict_proba(cible_scaled)



                    if proba[0][1] > threshold + threshold/3:
                        st.image('image/niv5.png')
                        st.write('big rain')
                    elif proba[0][1] >= threshold:
                        st.image('image/niv4.png')
                        st.write('rain')
                    elif proba[0][1] > threshold/2:
                        st.image('image/niv3.png')
                        st.write('cloudy')
                    elif proba[0][1] > threshold/3:
                        st.image('image/niv2.png')
                        st.write('little cloudy')
                    else :
                        st.image('image/niv1.png')
                        st.write('plain sun')

                with confidence:
                    acc = np.round(accuracy_score(y_test, y_pred), 2)

                    st.write('Confidence Rate :')
                    st.write(acc)



            # with seuil:
            #     st.write(proba)


                # Threshold = st.slider(
                #     label = 'Probability decision',
                #     min_value = 0.0,
                #     max_value = 1.0,
                #     value = threshold,
                #     step = 100.0,
                #     help = 'you can change the threshold, depending on your needs.',
                # )

            if st.button('More statistiques ?', type = 'secondary'):

                with st.container(border = False):

                    classif, graph = st.columns(2, border = True)

                    with classif:

                        st.table(classification_report(y_test, y_pred, output_dict = True))

                    with graph:

                        with st.spinner('Calcul des métriques...'):

                            f1_1 = []
                            roc_1 = []
                            precision1 = []
                            recall1 = []
                            accuracy = []

                            for i in np.linspace(0, 1, 999):
                                seuil = i
                                y_pred = (y_pred_prob1 > i).astype("int32")
                                accuracy.append(accuracy_score(y_test, y_pred))
                                f1_1.append(f1_score(y_test, y_pred))
                                roc_1.append(roc_auc_score(y_test, y_pred))
                                precision1.append(precision_score(y_test, y_pred))
                                recall1.append(recall_score(y_test, y_pred))

                            plt.figure(figsize = (20, 5))

                            plt.subplot(121)
                            plt.plot(accuracy, label = 'accuracy')
                            plt.plot(f1_1, label = 'f1')
                            plt.plot(roc_1, label = 'roc-auc')
                            plt.plot(precision1, label = 'precision')
                            plt.plot(recall1, label = 'recall')
                            plt.title(f'Evolution des metriques en fonction du seuil pour {city}')
                            plt.legend()

                            st.pyplot(plt)




            st.write('ok')
