import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


def inzilization(population_size, n_vertex):
    population = [list(np.random.permutation(n_vertex)) for i in range(population_size)]
    return population


def data_loader(name):
    with open(name + ".tsp") as f:
        lines = f.readlines()
    n_vertex = int(lines[3].split()[2])
    coordinates = []
    for i in range(n_vertex):
        num, x, y = lines[6 + i].split()
        coordinates.append((float(x), float(y)))

    graph = [[int(np.sqrt(
        np.power(coordinates[i][0] - coordinates[j][0], 2) + np.power(coordinates[i][1] - coordinates[j][1], 2)) + 0.5)
              for j in range(n_vertex)] for i in range(n_vertex)]

    with open(name + ".opt.tour") as f:
        lines = f.readlines()

    opt_tour = [int(lines[5 + i].split()[0]) - 1 for i in range(n_vertex)]

    return graph, opt_tour, n_vertex, coordinates


def data_loader_new(name):
    with open(name + ".tsp") as f:
        lines = f.readlines()
    n_vertex = int(lines[3].split()[2])
    coordinates = []
    for i in range(n_vertex):
        num, x, y = lines[6 + i].split()
        coordinates.append((float(x), float(y)))

    graph = [[int(np.sqrt(
        np.power(coordinates[i][0] - coordinates[j][0], 2) + np.power(coordinates[i][1] - coordinates[j][1], 2)) + 0.5)
              for j in range(n_vertex)] for i in range(n_vertex)]

    return graph, n_vertex, coordinates


def distance(graph, a, b):
    # print(a, b)
    return graph[a][b]


def tsp(graph):
    n = len(graph[0])

    # dp[i][j] stores minimum cost of traveling from
    dp = [[0 for x in range(n)] for y in range(2 ** n)]

    max_val = float("inf")

    for subset in range(1, 2 ** n):
        for j in range(n):
            dp[subset][j] = max_val
            if subset & (1 << j):
                if subset == (1 << j):
                    dp[subset][j] = graph[j][0]

                else:
                    for k in range(n):
                        if subset & (1 << k):
                            dp[subset][j] = min(dp[subset][j], graph[j][k] +
                                                dp[subset ^ (1 << j)][k]);

    min_cost = max_val;
    for j in range(n):
        min_cost = min(min_cost, dp[2 ** n - 1][j] + graph[j][0])
    return min_cost


def generate_graph(n, cnt):
    graph = [[0 for j in range(n)] for i in range(n)]

    # Generate random weights for the graph
    for i in range(n):
        for j in range(i):
            if i != j:
                graph[i][j] = random.randint(1, cnt)
                graph[j][i] = graph[i][j]

    return graph


def fitness(graph, item, n):
    f = distance(graph, item[-1], item[0])

    for i in range(n - 1):
        f += distance(graph, item[i], item[i + 1])

    return 1 / f


def length(graph, item, n):
    f = distance(graph, item[-1], item[0])
    for i in range(n - 1):
        f += distance(graph, item[i], item[i + 1])
    return f


def length_PDX(item, start, n):
    # print(start, item[0])
    f = distance(start, item[0])
    for i in range(n - 1):
        f += distance(item[i], item[i + 1])
    return f


def jaccard_distance(set_A, set_cluster):
    min_distance = float('inf')
    for set_B in set_cluster:
        jaccard_dist = 1 - len(set_A.intersection(set_B)) / len(set_A.union(set_B))
        if jaccard_dist < min_distance:
            min_distance = jaccard_dist
#     print(min_distance)
    return min_distance


def set_format(el):
    n = len(el)
    return set([(el[i % n], el[(i + 1) % n]) for i in range(n)])


def jaccard_distance_indiv(set_A, set_B):
    return 1 - len(set_A.intersection(set_B)) / len(set_A.union(set_B))


def my_centre(clust):
    cent = clust[0]
    min_dist = 1000000
    for el1 in clust:
        dist = 0
        for el2 in clust:
            dist += jaccard_distance_indiv(el1, el2)
        if min_dist > dist:
            min_dist = dist
            cent = el1
    return cent


def my_distance(el, mtr):
    dist = 0
    for i in range(len(el) - 1):
        dist += mtr[el[i]][el[i+1]]
        dist += mtr[el[i+1]][el[i]]
    dist += mtr[el[0]][el[len(el) - 1]]
    dist += mtr[el[len(el) - 1]][el[0]]
#     print(min_distance)
#     print(dist)
    return 1 - dist


def my_matr(n, clust):
    mtr = [[0]*n for i in range(n)]
    cnt = 1/(len(clust)*n)
    for el in clust:
        for i in range(len(el) - 1):
            mtr[el[i]][el[i+1]] += cnt
            mtr[el[i+1]][el[i]] += cnt
        mtr[el[0]][el[len(el) - 1]] += cnt
        mtr[el[len(el) - 1]][el[0]] += cnt
    return mtr


def zero_fit(set_A, set_cluster):
    return 0
