import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
from tqdm import tqdm
import get_tree as gt
import igraph as ig


def clean_genre(original_genres, keep_genre):
    """
    only keep the genres in the original_genre which belongs to keep_genre
    :param original_genre: the 2D list genre
    :param keep_genre: genre need to be kept
    :return: new 2D list
    """
    new_genre_list = []
    for row in tqdm(original_genres):
        row_keep = []
        for index in range(len(row)):
            if row[index] in keep_genre:
                row_keep.append(row[index])
        new_genre_list.append(row_keep)
    return new_genre_list


def get_one_hot(genres, genre_list):
    """
    one-hot code the genre and represent the genres with the code
    :param genres: the original 2D list genres
    :param genre_list: the unique genre sorted by f
    :return: 2D one-hot represented genres
    """
    # use one hot to represent the genres
    one_hot = pd.get_dummies(genre_list)
    # get the corresponding genre name
    # use vector to represent the genre
    genres_digit = []
    for row in tqdm(genres):
        digit = np.zeros((len(genre_list),), dtype=int)
        for name in row:
            digit += np.array(one_hot[name])
        genres_digit.append(list(digit))
    return genres_digit


# clear text
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(int(line_value[i]))
                    s += " "
                else:
                    s += str(int(line_value[i]))
                    s += "\n"
            f.write(s)

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
genres_selected = all_genres_df[all_genres_df['Count'] >= 400]
# delete the genres not contained in the genres_selected
frequent_genres = list(genres_selected.sort_values(by=['Count'], ascending=False)['Genre'])
genres_h_f = clean_genre(genres, frequent_genres)
# delete the 0 genre rows
movies['genre_new_1'] = genres_h_f
movies_new_1 = movies[~(movies['genre_new_1'].str.len() == 0)]
# get one hot represent genres
genres_digit = get_one_hot(list(movies_new_1['genre_new_1']), frequent_genres)
# get the sorted tree node after generating the tree
tree_node, tree_graph = gt.get_tree(genres_digit, frequent_genres, 15)
# plot
layout = tree_graph.layout_lgl()
ig.drawing.plot(tree_graph, "data/treeimg/15node_1.png", layout=layout, bbox=(1100, 1100), margin=(80, 80, 80, 80))
# delete the genres not contained in the tree node
genres_in_tree = clean_genre(list(movies_new_1['genre_new_1']), tree_node)
# delete the 0 genre rows
movies_new_1['genre_new_2'] = genres_in_tree
movies_new_2 = movies_new_1[~(movies_new_1['genre_new_2'].str.len() == 0)]
node_digit = get_one_hot(list(movies_new_2['genre_new_2']), tree_node)
check_tree_node, check_tree_graph = gt.get_tree(node_digit, tree_node, 15)
# plot
ig.drawing.plot(check_tree_graph, "data/treeimg/15node_2.png", layout=layout, bbox=(1100, 1100), margin=(80, 80, 80, 80))
# process the plot
movies_new_2['clean_plot'] = movies_new_2['plot'].apply(lambda x: preprocess_text(x))
# write to txt
# randomly split the data into two parts: training and testing
X_test = []
X_train = []
y_test = []
y_train = []
np.random.seed(666)
plot_list = list(movies_new_2['clean_plot'])
for i in range(len(node_digit)):
    if np.random.random() > 0.1:
        X_train.append(plot_list[i])
        y_train.append(node_digit[i])
    else:
        X_test.append(plot_list[i])
        y_test.append(node_digit[i])
x_write(X_test, "data/processed/X_test.txt")
x_write(X_train, "data/processed/X_train.txt")
y_write(y_train, "data/processed/y_train.txt")
y_write(y_test, "data/processed/y_test.txt")
print("done")


















































