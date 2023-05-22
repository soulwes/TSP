import numpy as np
from copy import copy, deepcopy
import collections
import networkx as nx
from Functions import fitness
import warnings
warnings.filterwarnings("ignore")


def rox(graph, item1, item2, n):
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)

    if a > b:
        a, b = b, a
    if a == b and b == 0:
        b += 1
    if a == b and b == n:
        a -= 1

    insert1 = item1[a:b]
    insert1_values = set(insert1)
    insert2 = []

    for i in range(n):
        if item2[i] not in insert1_values:
            insert2.append(item2[i])

    best_offspring = insert1.copy()
    best_offspring.extend(insert2)

    best_offspring_value = fitness(graph, best_offspring, n)
    # print(best_offspring_value)
    insert2 = collections.deque(insert2)

    for i in range(len(insert2) - 1):
        challanger = insert1.copy()
        insert2.rotate(1)
        challanger.extend(list(insert2))
        challanger_value = fitness(graph, challanger, n)

        if challanger_value > best_offspring_value:
            best_offspring_value = challanger_value
            best_offspring = challanger.copy()

    return best_offspring


def dpx(graph, item1, item2, n):
    edges1 = set()
    edges2 = set()

    for i in range(n):
        if item1[i % n] < item1[(i + 1) % n]:
            edges1.add((item1[i % n], item1[(i + 1) % n]))
        else:
            edges1.add((item1[(i + 1) % n], item1[i % n]))
        if item2[i % n] < item2[(i + 1) % n]:
            edges2.add((item2[i % n], item2[(i + 1) % n]))
        else:
            edges2.add((item2[(i + 1) % n], item2[i % n]))

    edges_both = edges1.intersection(edges2)

    list_edges = []
    for i in edges_both:
        list_edges.append(i)

    G = nx.Graph()

    G.add_edges_from(list_edges)
    G.add_nodes_from(item1)

    if G.number_of_edges() == n:
        return deepcopy(item1)

    subsequences = []

    for i in nx.connected_components(G):

        start_end = []

        for j in i:

            if G.degree[j] <= 1:
                start_end.append(j)

        if len(start_end) == 2:
            subsequences.append(nx.shortest_path(G, start_end[0], start_end[1]))
        else:
            subsequences.append(start_end)

    offspring = []
    to_add = set([i for i in range(len(subsequences))])

    first = np.random.randint(0, len(subsequences))

    offspring.extend(subsequences[first])

    to_add.remove(first)

    for i in range(len(subsequences) - 1):
        best = -1
        best_value = -1
        inverse = False
        for j in to_add:
            if best_value == -1 or graph[offspring[-1]][subsequences[j][0]] < best_value:
                best = j
                best_value = graph[offspring[-1]][subsequences[j][0]]
                inverse = False

            if best_value == -1 or graph[offspring[-1]][subsequences[j][-1]] < best_value:
                best = j
                best_value = graph[offspring[-1]][subsequences[j][-1]]
                inverse = True

        if not inverse:
            offspring.extend(subsequences[best])
            to_add.remove(best)
        else:
            offspring.extend(subsequences[best][::-1])
            to_add.remove(best)

    return offspring
