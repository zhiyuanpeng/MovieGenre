# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix representation of the graph
import pandas as pd
import numpy as np
import sys  # Library for INT_MAX
import igraph as ig
from tqdm import tqdm
from itertools import combinations


class MaxSpanningGraph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

        # A utility function to print the constructed MST stored in parent[]

    def print_mst(self, parent):
        print("Edge \tWeight")
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            # A utility function to find the vertex with

    def save_mst(self, parent):
        edges = []
        edge_weights = []
        for i in range(1, self.V):
            edges.append((str(parent[i]), str(i)))
            edge_weights.append(self.graph[i][parent[i]])
        return edges, edge_weights

    # maximum distance value, from the set of vertices
    # not yet included in shortest path tree
    def max_key(self, key, mstSet):
        # Initilaize max value
        max_value = float('-inf')
        for v in range(self.V):
            if key[v] > max_value and mstSet[v] == False:
                max_value = key[v]
                max_index = v
        return max_index, max_value
        # Function to construct and print MST for a graph

    # represented using adjacency matrix representation
    def prim_mst(self):
        # Key values used to pick maximum weight edge in cut
        key = [float('-inf')] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1  # First node is always the root of
        score_sum = 0
        for cout in range(self.V):
            # Pick the maximum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u, score = self.max_key(key, mst_set)
            score_sum += score
            # Put the minimum distance vertex in
            # the shortest path tree
            mst_set[u] = True
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the longest path tree
            for v in range(self.V):
                # graph[u][v] is non zero only for adjacent vertices of m
                # mst_set[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > key[v] and mst_set[v] == False:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        return self.save_mst(parent), score_sum


def get_co_score(list_name):

    toxic_comments_labels = np.array(list_name)

    # get each label's count
    count_list = []
    for i in range(toxic_comments_labels.shape[1]):
        count_list.append(np.sum(toxic_comments_labels[:, i]))

    # get co_occurrence count
    co_count_matrix = np.zeros((toxic_comments_labels.shape[1], toxic_comments_labels.shape[1]))
    for i in range(co_count_matrix.shape[0]):
        for j in range(i + 1, co_count_matrix.shape[1]):
            co_count_matrix[i, j] = np.sum(toxic_comments_labels[:, i] * toxic_comments_labels[:, j])

    # get co_score
    co_score = np.zeros_like(co_count_matrix)
    for i in range(co_score.shape[0]):
        for j in range(i + 1, co_score.shape[1]):
            co_score[i, j] = co_count_matrix[i, j] / np.minimum(count_list[i], count_list[j])
            co_score[j, i] = co_score[i, j]
    return co_score


def mst_k(list_name, k):
    co_score = get_co_score(list_name)
    index_list = [i for i in range(len(list_name[0]))]
    # all_possible is a list of tuples
    all_possible = combinations(index_list, k)
    edges_cache = []
    weights_cache = []
    score_cache = []
    for one_possible in tqdm(all_possible):
        co_score_k = np.zeros((k, k))
        row = 0
        for i in one_possible:
            column = 0
            for j in one_possible:
                co_score_k[row, column] = co_score[i, j]
                column += 1
            row += 1
        max_spanning = MaxSpanningGraph(co_score_k.shape[0])
        max_spanning.graph = co_score_k
        (edges, weights), score_sum = max_spanning.prim_mst()
        edges_cache.append(edges)
        weights_cache.append(weights)
        score_cache.append(score_sum)
    max_score = max(score_cache)
    for i in range(len(score_cache)):
        if score_cache[i] == max_score:
            return edges_cache[i], weights_cache[i], max_score


def edges_selected(threshold, edges, weights):
    """
    select the edge the weight of which is bigger than threshold
    :param threshold:
    :param edges:
    :param weights:
    :return:
    """
    edges_new = []
    weights_new = []
    for index in range(len(weights)):
        if weights[index] >= threshold:
            edges_new.append(edges[index])
            weights_new.append(weights[index])
    unique_v = set(sum(edges_new, ()))
    return edges_new, weights_new, list(unique_v)


def get_tree(list_name, k):
    edges, edge_weights, max_score = mst_k(list_name, k)
    # plot the graph
    tree_graph = ig.Graph()
    tree_graph.add_vertices([str(i) for i in range(k)])
    tree_graph.add_edges(edges)
    tree_graph.vs["label"] = tree_graph.vs["name"]
    layout = tree_graph.layout_kamada_kawai()
    ig.drawing.plot(tree_graph, layout=layout)
    print(max_score)
