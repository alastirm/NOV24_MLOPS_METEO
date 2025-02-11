import warnings

import pandas as pd

from sklearn.pipeline import Pipeline

from preprocess_sunshine import sunshine_transformer
from preprocess_raintomorrow import raintomorrow_transformer
from preprocess_date import preprocess_date_transformer
from preprocess_temp import preprocess_temp
from preprocess_rainfall_raintoday import rainfall_raintoday_transformer
from preprocess_wind import wind_speed_transformer, wind_dir_transformer
from preprocess_humidity_pressure import VoisinageNAImputer
from preprocess_evaporation import evaporation_transformer
from preprocess_cloud import cloud_transformer


from modelisation import modelisation

def preprocessing(url_data : str = '../data/weatherAUS.csv', city : str = 'Sydney'):

    warnings.filterwarnings('ignore')

    df = pd.read_csv(url_data)

    print(df['Location'].unique(), end = '\n\n')

    while True:

        city = input('Quelle ville choisissez vous ?\n ---->\n\n').capitalize()

        if city in df['Location'].unique():
            break  # La ville est valide, on sort de la boucle
        else:
            print(f"La ville '{city}' n'est pas dans la liste. \n\n Veuillez choisir une ville parmi celles disponibles.")

    df = df[df['Location'].isin([city])]



    columns_to_drop = [col for col in df.columns if df[col].isna().mean() > 0.30]

    print('colonne drop', columns_to_drop)
    print('colonne du dataframe', df.columns)

    transformers = []

    if 'RainTomorrow' not in columns_to_drop:
        transformers.append(('raintomorrow_transformer', raintomorrow_transformer()))

    if 'Evaporation' not in columns_to_drop:
        transformers.append(('evaporation_transformer', evaporation_transformer()))

    if 'Sunshine' not in columns_to_drop:
        transformers.append(('sunshine_transformer', sunshine_transformer()))

    if 'Date' not in columns_to_drop:
        transformers.append(('date_transformer', preprocess_date_transformer()))

    for temp_col in ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']:
        if temp_col not in columns_to_drop:
            transformers.append((f'{temp_col.lower()}_transformer', preprocess_temp(col_select=temp_col)))

    for cloud_col in ['Cloud9am', 'Cloud3pm']:
        if cloud_col not in columns_to_drop:
            transformers.append((f'{cloud_col.lower()}_transformer', cloud_transformer(col_select=cloud_col)))

    for wind_col in ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']:
        if wind_col not in columns_to_drop:
            transformers.append((f'{wind_col.lower()}_transformer', wind_speed_transformer(col_select=wind_col)))

    for wind_dir_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if wind_dir_col not in columns_to_drop:
            transformers.append((f'{wind_dir_col.lower()}_transformer', wind_dir_transformer(col_select=wind_dir_col)))

    if 'Rainfall' not in columns_to_drop:
        transformers.append(('rainfall_transformer', rainfall_raintoday_transformer(city=city)))

    for humidity_col in ['Humidity9am', 'Humidity3pm']:
        if humidity_col not in columns_to_drop:
            transformers.append((f'{humidity_col.lower()}_imputer', VoisinageNAImputer(column=humidity_col)))

    for pressure_col in ['Pressure9am', 'Pressure3pm']:
        if pressure_col not in columns_to_drop:
            transformers.append((f'{pressure_col.lower()}_imputer', VoisinageNAImputer(column=pressure_col)))

    df_transformed = Pipeline(transformers).fit_transform(df)

    df = df.drop(columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'] + columns_to_drop)
    df = df.set_index('Date')

    df.dropna(inplace = True)
    print('apres', df.info())

    # df.to_csv('../data_saved/data_mat2.csv')
    modelisation(df)









if __name__ == "__main__":
    print(preprocessing())
