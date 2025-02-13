from modelisation import modelisation
from preprocess import preprocessing

import pandas as pd

def main():

    url_data = '../data/weatherAUS.csv'


    df = pd.read_csv(url_data)

    print(df['Location'].unique(), end = '\n\n')

    while True:

        city = input('Quelle ville choisissez vous ?\n ---->\n\n').capitalize()

        if city in df['Location'].unique():
            break
        else:
            print(f"La ville '{city}' n'est pas dans la liste. \n\n Veuillez choisir une ville parmi celles disponibles.")

    df_city = df[df['Location'].isin([city])]

    df_preproc = preprocessing(df_city, city = city)
    modelisation(df_preproc)



if __name__ == "__main__":
    main()
