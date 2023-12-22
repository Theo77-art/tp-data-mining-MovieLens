import pandas as pd


def get_data(filename):
    chemin_fichier_csv = 'data/' + filename
    dataframe = pd.read_csv(chemin_fichier_csv)
    return dataframe


filenames = ["links.csv", "movies.csv", "ratings.csv", "tags.csv"]

# Dictionnaire pour stocker les DataFrames
dfs = {}

# Charger chaque fichier dans un DataFrame séparé
for filename in filenames:
    try:
        dfs[filename.split('.')[0]] = get_data(filename)
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")

print(dfs)