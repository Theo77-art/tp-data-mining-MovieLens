import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------Lecture des fichiers------------------#


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

# ---------------Etape de reconnaissance------------------#

# Afficher les premières lignes, les infos et les statistiques descriptives de chaque DataFrame
for df_name, df in dfs.items():
    print(f"{df_name} - Premières lignes:")
    print(df.head())
    print(f"{df_name} - Info:")
    print(df.info())
    print(f"{df_name} - Statistiques descriptives:")
    print(df.describe())

# ---------------nettoyage et prétraitement------------------#

#### Pour le DataFrame links ####
# Remplacer les valeurs manquantes dans tmdbId par -1
dfs['links']['tmdbId'].fillna(-1, inplace=True)

# Vérifier l'unicité des IDs
assert dfs['links']['movieId'].is_unique
assert dfs['links']['imdbId'].is_unique

#### Pour le DataFrame movies ####
# Supprimer les doublons basés sur movieId
dfs['movies'].drop_duplicates(subset='movieId', inplace=True)

# Nettoyer les espaces supplémentaires dans le titre
dfs['movies']['title'] = dfs['movies']['title'].str.strip()

# Diviser les genres en listes et les convertir en minuscules
dfs['movies']['genres'] = dfs['movies']['genres'].str.lower().str.split('|')

#### Pour le DataFrame ratings ####
# Supprimer les doublons basés sur la combinaison de userId et movieId
dfs['ratings'].drop_duplicates(subset=['userId', 'movieId'], inplace=True)

# Vérifier que les notes sont dans l'échelle attendue et traiter les valeurs hors échelle
dfs['ratings'] = dfs['ratings'][(dfs['ratings']['rating'] >= 0.5) & (dfs['ratings']['rating'] <= 5)]

# Convertir les timestamps en format date/heure standard
dfs['ratings']['timestamp'] = pd.to_datetime(dfs['ratings']['timestamp'], unit='s')

#### Pour le DataFrame tags ####
# Supprimer les doublons basés sur la combinaison de userId, movieId et tag
dfs['tags'].drop_duplicates(subset=['userId', 'movieId', 'tag'], inplace=True)
# Nettoyer les tags: enlever les espaces supplémentaires et convertir en minuscules
dfs['tags']['tag'] = dfs['tags']['tag'].str.lower().str.strip()
# Convertir les timestamps en format date/heure standard
dfs['tags']['timestamp'] = pd.to_datetime(dfs['tags']['timestamp'], unit='s')

# Vérifier la cohérence des IDs de film à travers les différents DataFrames
# assert dfs['movies']['movieId'].isin(dfs['ratings']['movieId']).all()
# assert dfs['movies']['movieId'].isin(dfs['tags']['movieId']).all()

# Vérifier la cohérence des IDs d'utilisateur à travers les différents DataFrames
# assert dfs['ratings']['userId'].isin(dfs['tags']['userId']).all()

"""
Référentiel pour le DataFrame links :
- movieId (int64) : Identifiant unique pour chaque film.
- Action : Vérifier l'unicité, pas de traitement spécial requis.
imdbId (int64) : Identifiant correspondant à la base de données IMDb.
Action : Vérifier l'unicité, pas de traitement spécial requis.
tmdbId (float64) : Identifiant correspondant à la base de données TMDb.
Action : Remplacer les valeurs manquantes par une valeur indicative (par exemple, 0 ou -1) ou les supprimer si non utilisées.
Référentiel pour le DataFrame movies :
movieId (int64) : Identifiant unique pour chaque film.
Action : Vérifier l'unicité et l'absence de doublons.
title (object) : Titre du film.
Action : Nettoyer les espaces supplémentaires, corriger les erreurs d'orthographe si possible.
genres (object) : Genres du film séparés par des barres verticales.
Action : Diviser en listes de genres, normaliser les noms de genres (par exemple, en minuscules).
Référentiel pour le DataFrame ratings :
userId (int64) : Identifiant unique de l'utilisateur.
Action : Vérifier l'unicité dans le contexte de chaque film.
movieId (int64) : Identifiant du film noté.
Action : Vérifier l'existence correspondante dans le DataFrame movies.
rating (float64) : Note attribuée par l'utilisateur.
Action : Vérifier que les valeurs sont dans l'échelle attendue (0.5 à 5). Traiter ou supprimer les valeurs hors échelle.
timestamp (int64) : Horodatage de la note.
Action : Convertir en format date/heure standard si nécessaire.
Référentiel pour le DataFrame tags :
userId (int64) : Identifiant unique de l'utilisateur.
Action : Vérifier l'unicité dans le contexte de chaque combinaison de film et de tag.
movieId (int64) : Identifiant du film tagué.
Action : Vérifier l'existence correspondante dans le DataFrame movies.
tag (object) : Tag ou mot-clé attribué au film.
Action : Nettoyer les espaces supplémentaires, convertir en minuscules, et éventuellement regrouper les tags similaires.
timestamp (int64) : Horodatage du tag.
Action : Convertir en format date/heure standard si nécessaire.
Actions de nettoyage et de prétraitement :
Supprimer les doublons : Pour chaque DataFrame, supprimez les lignes en double.
Gérer les valeurs manquantes : Décidez d'une stratégie pour traiter ou supprimer les valeurs manquantes.
Vérifier la cohérence : Assurez-vous que les identifiants de film et d'utilisateur sont cohérents à travers les différents DataFrames.
Normaliser les données : Standardisez les formats des chaînes de caractères, les échelles de notation, etc.
Convertir les formats : Assurez-vous que les dates/heures sont dans un format utilisable et cohérent.
"""

# ---------------algorithme de recommandation simple------------------#

# Creation d'une matrice de notation utilisateur/film - lignes = utilisateurs, colonnes = films
rating_matrix = dfs['ratings'].pivot_table(index='userId', columns='movieId', values='rating')

# Normalisation des notes
mean_ratings = rating_matrix.mean(axis=1)
rating_normalized = (rating_matrix.T - mean_ratings).T

# Remplacer les valeurs NaN par 0
rating_normalized = rating_normalized.fillna(0)

# Calculer la similarité cosinus
item_similarity = cosine_similarity(rating_normalized.T)

# Créer un DataFrame à partir de la matrice de similarité
user_predicted_ratings = rating_normalized.dot(item_similarity) / item_similarity.sum(axis=1)


def recommend_movies(user_id, user_predicted_ratings, movies_df, original_ratings_df, num_recommendations=5):
    # Obtenir et trier les prédictions de l'utilisateur
    user_row_number = user_id - 1  # L'ID d'utilisateur commence à 1
    sorted_user_predictions = user_predicted_ratings.iloc[user_row_number].sort_values(ascending=False)

    # Obtenir les données de l'utilisateur et fusionner avec les données des films
    user_data = original_ratings_df[original_ratings_df.userId == user_id]
    user_full = (user_data.merge(movies_df, how='left', on='movieId').
                 sort_values(['rating'], ascending=False)
                 )

    # Recommander les films que l'utilisateur n'a pas encore vu
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='movieId',
                             right_on='movieId').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:num_recommendations, :-1]
                       )

    return user_full, recommendations


# ---------------Test de l'algo------------------#

user_full, recommendations = recommend_movies(1, pd.DataFrame(user_predicted_ratings), dfs['movies'], dfs['ratings'])

# Afficher les résultats
print("Films déjà notés par l'utilisateur:")
print(user_full.head())  # Afficher les films déjà notés par l'utilisateur

print("\nFilms recommandés:")
print(recommendations)  # Afficher les films recommandés

#Faire un Rapport avec explication des choix fait