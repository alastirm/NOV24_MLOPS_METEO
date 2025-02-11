import sys
import os

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/mathieu/code/MathieuAmacher/datascientest/NOV24-BDS-METEO/src2')
import preprocess



df = pd.read_csv('data/weatherAUS.csv')


with st.container(border = True):

    city = st.selectbox(label = 'Select a City',
                    options = sorted(df['Location'].unique()),
                    index = None,
                    label_visibility="visible",
                    help = 'get a listed city',
                    placeholder = 'No city selected yet'
                    )

if city :

    df_city = df[df['Location'] == city]

    preprocess.preprocessing(url_data = 'data/weatherAUS.csv', city = city)



    with st.container(border = False):

        wind1, wind2 = st.columns(2, border = True)

        with wind1:
            st.write("How about Wind Speed ?")

            speed1, speed2, speed3 = st.columns(3)

            with speed1:
                windspeed9am = st.number_input(label = 'at 9:00 am',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = df_city['WindSpeed9am'].mode()[0].astype(float),
                                help = 'Look at your anemometer at 9 in the morning',
                                )

            with speed2:
                windspeed3pm = st.number_input(label = 'at 3:00 pm',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = df_city['WindSpeed3pm'].mode()[0].astype(float),
                                help = 'Look at your anemometer at 3 in the afternoon',
                                )

            with speed3:
                windgustspeed = st.number_input(label = 'Gust',
                                min_value = 0.0,
                                max_value = 200.0,
                                value = df_city['WindGustSpeed'].mode()[0].astype(float),
                                help = 'Look at your anemometer for the fastest wind speed of the day',
                                )

        with wind2:
            st.write("How about Wind Direction ?")

            dir1, dir2, dir3 = st.columns(3)

            with dir1:
                winddir9am = st.selectbox(label = '9:00 am',
                                        options = sorted(df['WindDir9am'].dropna().unique()),
                                        help = 'Look at your compass at 9:00 am',
                                        placeholder = 'compass..',
                                        )

            with dir2:
                winddir3pm = st.selectbox(label = '3:00 pm',
                                        options = sorted(df['WindDir3pm'].dropna().unique()),
                                        help = 'Look at your compass at 3:00 pm',
                                        placeholder = 'compass..',
                                        )

            with dir3:
                windgustdir = st.selectbox(label = 'Gust',
                                        options = sorted(df['WindGustDir'].dropna().unique()),
                                        help = 'Look at your when the gust blow',
                                        placeholder = 'compass..',
                                        )


    with st.container(border = False):

        temp_mm, temp_h = st.columns(2, border = True)

        with temp_mm:

            st.write("Select the min and max temperature")

            MinTemp, MaxTemp = st.select_slider(
                label = 'slide it',
                options = np.round(np.arange(0, 50, 0.1), 2),
                value=(df_city['MinTemp'].mode()[0].astype(float), df_city['MaxTemp'].mode()[0].astype(float)),
            )

        with temp_h:

            st.write('What is the temperature at ?')

            temp9, temp3 = st.columns(2, border = False)

            with temp9:
                Temp9am = st.number_input(label = '9:00 am',
                                min_value = 0.0,
                                max_value = 50.0,
                                value = df_city['Temp9am'].mode()[0].astype(float),
                                help = 'Look at your thermometer at 9 in the morning',
                                )

            with temp3:
                Temp3pm =  st.number_input(label = '3:00 pm',
                                min_value = 0.0,
                                max_value = 50.0,
                                value = df_city['Temp3pm'].mode()[0].astype(float),
                                help = 'Look at your thermometer at 3 in the afternoon',
                                )

    with st.container(border = False):

        press, cloud = st.columns(2, border = True)

        with press:

            st.write('Pressure at 9:00 am and 3:00 pm')

            press9, press3 = st.columns(2, border = False)

            with press9:

                Pressure9am = st.number_input(label = 'at 9:00 am',
                                    min_value = 970.0,
                                    max_value = 1050.0,
                                    value = df_city['Pressure9am'].mode()[0].astype(float),
                                    help = 'Look at your barometer at 9 in the morning',
                                    )

            with press3:

                Pressure3pm = st.number_input(label = 'at 3:00 pm',
                                    min_value = 970.0,
                                    max_value = 1050.0,
                                    value = df_city['Pressure3pm'].mode()[0].astype(float),
                                    help = 'Look at your barometer at 3 in the afternoon',
                                    )

        with cloud:

            st.write('Give the nebulosity ?')

            cloud9, cloud3 = st.columns(2, border = False)

            with cloud9:

                Cloud9am = st.selectbox(label = '9:00 am',
                                options = np.arange(0, 10, 1),
                                help = 'Look at the sky',
                                placeholder = '...',
                                )

            with cloud3:

                Cloud3pm = st.selectbox(label = '3:00 pm',
                                options = np.arange(0, 10, 1),
                                help = 'Look at the sky',
                                placeholder = '...',
                                )

    with st.container(border = False):

        hum, evap= st.columns(2, border = True)

        with hum:

            st.write('What is the humidity ?')


            hum9, hum3 = st.columns(2, border = False)

            with hum9:

                Humidity9am = st.slider(
                    label = '9:00 am',
                    min_value = 0.0,
                    max_value = 100.0,
                    value = df_city['Humidity9am'].mean(),
                    step = 1.0,
                    help = 'give the humidity at 9:00 am'
                )

            with hum3:

                Humidity3pm = st.slider(
                    label = '3:00 pm',
                    min_value = 0.0,
                    max_value = 100.0,
                    value = df_city['Humidity3pm'].mode()[0].astype(float),
                    step = 1.0,
                    help = 'give the humidity at 3:00 pm'
                )

        with evap:

            st.write('Do you know the amount of evaporation ?')

            Evaporation = st.slider(
                label = 'in mm',
                min_value = 0.0,
                max_value = 21.0,
                value = 5.0,
                step = 1.0,
                help = 'hard help yourself',
            )



    with st.container(border = True):

        sunsh, raintoday = st.columns(2, border = False)

        with sunsh:

            Sunshine = st.number_input(label = 'How many hours of sun today ?',
                                        min_value = 0.0,
                                        max_value = 14.5,
                                        step = 0.25,
                                        value = 5.0,
                                        help = 'that is obsvious',
                                    )

        with raintoday :

            RainToday = st.checkbox(label = 'Does it rain today ?',
                                    value = False,
            )

        if RainToday:


                Rainfall = st.number_input(label = 'How many minimeter of rain roday? ',
                                        min_value = 0.0,
                                        max_value = 371.0,
                                        value = df_city['Rainfall'].mean(),
                                        help = 'that is obsvious',
                                    )

    if st.button("Rain Tomorrow ?", type="primary"):
            st.write('city :', city)
            st.write('MinTemp :', MinTemp)
            st.write('MaxTemp :', MaxTemp)
            st.write('Rainfall', Rainfall)
            st.write('Evaporation', Evaporation)
            st.write('Sunshine', Sunshine)
            st.write('WindSpeed9am :', windspeed9am)
            st.write('WindSpeed3pm :', windspeed3pm)
            st.write('WindGustSpeed :', windgustspeed)
            st.write('WindDir9am :', winddir9am)
            st.write('WindDir3pm :', winddir3pm)
            st.write('WindGustDir :', windgustdir)

            st.write('Temp9am', Temp9am)
            st.write('Temp3pm', Temp3pm)
