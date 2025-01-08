# Importer les bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Fonction pour appliquer la médiane selon la localisation et la saison
def preprocess_temperatures_median_location(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]):
    median_values = df.groupby(["Season", "Location"])[columns].median()
    for col in columns:
        df[col] = df.set_index(["Season", "Location"])[col].fillna(median_values[col]).values
    return df

# Fonction pour appliquer la médiane selon le climat et la saison
def preprocess_temperatures_median_climate(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]):
    median_values = df.groupby(["Season", "Climate"])[columns].median()
    for col in columns:
        df[col] = df.set_index(["Season", "Climate"])[col].fillna(median_values[col]).values
    return df

# Fonction pour appliquer la moyenne des valeurs de la veille et du lendemain
def preprocess_temperature_mean(df, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]):
    for col in columns:
        df[col] = df[col].interpolate(method="linear")
    return df

# Appeler les fonctions de preprocessing
df_median_location = preprocess_temperatures_median_location(df.copy())
df_median_climate = preprocess_temperatures_median_climate(df.copy())
df_mean = preprocess_temperature_mean(df.copy())

# Vérification des méthodes
def verification_methodes(df, df_median1, df_median2, df_mean, columns=["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]):
    for col in columns:

        # Visualisation graphique
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=col, label="Original", color="firebrick", alpha=0.5)
        sns.histplot(data=df_median1, x=col, label="Médiane (Location)", color="blue", alpha=0.5)
        sns.histplot(data=df_median2, x=col, label="Médiane (Climate)", color="dodgerblue", alpha=0.5)
        sns.histplot(data=df_mean, x=col, label="Moyenne", color="green", alpha=0.5)
        plt.title(f"Distribution de {col}")
        plt.legend()
        plt.show()

        # Test de Kolmogorov-Smirnov
        ks_result_median1 = stats.ks_2samp(df[col].dropna(), df_median1[col].dropna())
        ks_result_median2 = stats.ks_2samp(df[col].dropna(), df_median2[col].dropna())
        ks_result_mean = stats.ks_2samp(df[col].dropna(), df_mean[col].dropna())
        print(f"Test KS pour {col} - Mediane par Season et Location : \n{ks_result_median1}\n")
        print(f"Test KS pour {col} - Mediane par Season et Climate : \n{ks_result_median2}\n")
        print(f"Test KS pour {col} - Moyenne de la veille et du lendemain : \n{ks_result_mean}\n\n")

    # Pourcentage de NaN
    print("Pourcentage de NaN apres traitement par la mediane par Season et Location :")
    print(df_median1[columns].isnull().mean() * 100, "\n")
    print("Pourcentage de NaN apres traitement par la mediane par Season et Climate :")
    print(df_median2[columns].isnull().mean() * 100, "\n")
    print("Pourcentage de NaN apres traitement par la moyenne de la veille et du lendemain :")
    print(df_mean[columns].isnull().mean() * 100, "\n")

# Appeler la fonction de vérification
verification_methodes(df, df_median_location, df_median_climate, df_mean)