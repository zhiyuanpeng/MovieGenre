import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("../Datasets/MovieSummaries/movie.metadata.tsv", sep='\t', header=None)
meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]

plots = []
with open("../Datasets/MovieSummaries/plot_summaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

# an empty list
genres = []

# extract genres
for i in movies['genre']:
    genres.append(list(json.loads(i).values()))

# get statistical info of genres
all_genres = sum(genres, [])
unique_genres = list(set(all_genres))
all_genres = nltk.FreqDist(all_genres)
# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
                              'Count': list(all_genres.values())})
genres_selected = all_genres_df[all_genres_df['Count'] >= 100]
# delete the genres not contained in the genres_selected
frequent_genres = list(genres_selected.sort_values(by=['Count'], ascending=False)['Genre'])
for row in tqdm(genres):
    for index in range(len(row) - 1):
        if row[index] not in frequent_genres:
            row.remove(row[index])
movies['genre_new'] = genres
# delete the all 0 genre
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
#
# g = all_genres_df.nlargest(columns="Count", n=50)
# plt.figure(figsize=(35, 45))
# ax = sns.barplot(data=genres_selected, x="Count", y="Genre")
# ax.set(ylabel='Count')
# plt.show()
print("done")

















































