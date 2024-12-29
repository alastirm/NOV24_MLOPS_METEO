import pandas as pd

# prérequis à l'utilisation de la fonction
# la colonne climate(minuscule) doit être ajoutée au df
# les na de la colonne RainTomorrow doivent être supprimer



# df = pd.read_csv('../data/weatherAUS.csv')
# df['climate'] = df['Location'].map(climate)
# df = df.dropna(subset = 'RainTomorrow')


def preproc_windgustspeed(df):

    '''création d'un dictionnaire, les clés sont un tuple (valeur de climate, valeur de Raintomorrow),
    toute les paires sont présentes en tant que clé, la valeur est la median de WindGustSpeed pour cette paire.
    puis on itere sur les lignes du df pour rechercher le na de la colonne WindGustSpeed
    on vérifie les valeurs climate et RainTomorrow de la ligne ou le na est présent
    on change la valeur de WingGustSpeed par la median'''


    dict_wgs = {}

    for i in df['climate'].unique():
        for j in df['RainTomorrow'].unique():
            dict_wgs[i, j] = (df[(df['climate'] == i) & (df['RainTomorrow'] == j)]['WindGustSpeed'].median())


    for index, i in df['WindGustSpeed'].items():
        if pd.isna(i):
            df.loc[index, 'WindGustSpeed']  = dict_wgs[df.loc[index, 'climate'], df.loc[index, 'RainTomorrow']]


    return df
