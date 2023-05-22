import numpy as np
from copy import copy, deepcopy
import collections
import networkx as nx
from sklearn.cluster import AffinityPropagation, DBSCAN
import random
from scipy.special import gamma
from sklearn.neighbors import KDTree
from collections import defaultdict
from tqdm import tqdm
from numpy import unique
from numpy import where
from matplotlib import pyplot
from itertools import combinations
from random import sample
from itertools import permutations
import random
import warnings
warnings.filterwarnings("ignore")


def pair_distance(item1, item2, n):
    edges1 = set([(item1[i % n], item1[(i + 1) % n]) for i in range(n)])
    edges2 = set([(item2[i % n], item2[(i + 1) % n]) for i in range(n)])

    return 1 - len(edges1.intersection(edges2)) / len(edges1.union(edges2))


def population_distance(population, n):
    distance_matrix = np.array([[pair_distance(population[i], population[j], n) for j in range(len(population))]
                                 for i in range(len(population))])
    return distance_matrix


def classic(population, distance_matrix):

    return [range(len(population))], [population]


def dbscan(population, distance_matrix, eps=0.3):

    # distance_matrix = np.matrix([[pair_distance(population[i], population[j], n) for j in range(len(population))]
    #                                  for i in range(len(population))])

    model = DBSCAN(eps=eps, metric="precomputed")  # 0.3

    yhat = model.fit_predict(distance_matrix)

    clusters = [[] for i in range(len(set(yhat)))]
    clusts_ = [[] for i in range(len(set(yhat)))]

    for i in range(len(population)):
        clusters[yhat[i]].append(population[i])
        clusts_[yhat[i]].append(i)

    return clusts_, clusters


def dbscan_test(population, distance_matrix, temp_best, best1, best2, eps=0.3):
    model = DBSCAN(eps=eps, metric="precomputed")  # 0.3

    yhat = model.fit_predict(distance_matrix)

    temp_yhat = []
    for i in range(len(population)):
        if i != best1 and i != best2:
            temp_yhat.append(yhat[i])
    temp_yhat = np.array(temp_yhat)

    clusters = unique(yhat)

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(temp_yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(temp_best[row_ix, 0], temp_best[row_ix, 1])
    pyplot.show()

    clusters = [[] for i in range(len(set(yhat)))]

    for i in range(len(population)):
        clusters[yhat[i]].append(population[i])
    print([len(clusters[i]) for i in range(len(set(yhat)))])
    return clusters