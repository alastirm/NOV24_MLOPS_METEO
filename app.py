import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('data/weatherAUS.csv')

with st.sidebar:
    st.title('MeteoStralia')




with st.container(border = True):

    city = st.selectbox(label = 'Select a City',
                    options = sorted(df['Location'].unique()),
                    index = None,
                    label_visibility="visible",
                    help = 'get a listed city',
                    placeholder = 'No city selected yet'
                    )

with st.container(border = True):

    wind1, wind2 = st.columns(2, border = True)

    with wind1:
        st.write("How about Wind Speed ?")

        speed1, speed2, speed3 = st.columns(3)

        with speed1:
            windspeed9am = st.number_input(label = 'at 9:00 am',
                            min_value = 0,
                            max_value = 200,
                            help = 'Look at your anemometer at 9 in the morning',
                            )

        with speed2:
            windspeed3pm = st.number_input(label = 'at 3:00 pm',
                            min_value = 0,
                            max_value = 200,
                            help = 'Look at your anemometer at 3 in the afternoon',
                            )

        with speed3:
            windgustspeed = st.number_input(label = 'Gust',
                            min_value = 0,
                            max_value = 200,
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
                                      help = 'Look at your compass at 9:00 am',
                                      placeholder = 'compass..',
                                      )

        with dir3:
            windgustdir = st.selectbox(label = 'Gust',
                                       options = sorted(df['WindGustDir'].dropna().unique()),
                                       help = 'Look at your when the gust blow',
                                       placeholder = 'compass..',
                                       )

if st.button("Rain Tomorrow ?", type="primary"):
    st.write('city :', city)
    st.write('WindSpeed9am :', windspeed9am)
    st.write('WindSpeed3pm :', windspeed3pm)
    st.write('WindGustSpeed :', windgustspeed)
    st.write('WindDir9am :', winddir9am)
    st.write('WindDir3pm :', winddir3pm)
    st.write('WindGustDir :', windgustdir)
