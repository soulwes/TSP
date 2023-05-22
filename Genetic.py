import numpy as np
from tqdm import tqdm
import warnings
from Functions import *
from Choice import choise
from Crossovers import *
from Mutation import *
from Clustering import *
warnings.filterwarnings("ignore")


def genetic_jaccard(n, graph, population_size, clasterization1, clust_dist, population1,
                    mutation_prob=0.4, alp=20, bet=20, max_iter=50, hunter_prob=0.25, balans=False, plot=False):
    population = population1.copy()
    population_distnces = [length(graph, population[i], n) for i in range(population_size)]
    routes = []
    print(len(population), "min:", min(population_distnces), "avg:", sum(population_distnces) / population_size, "max:",
          max(population_distnces))
    ans = []

    for iter in tqdm(range(max_iter)):
        routes = []
        distance_matrix = population_distance(population, n)
        clusts_, clusters = clasterization1(population, distance_matrix)
        for i, cluster in enumerate(clusters):

            route_cluster = []
            route_cluster.extend(cluster)

            mutations_counter = 0
            clust_matrix = distance_matrix[np.array(clusts_[i])[:, np.newaxis], np.array(clusts_[i])]
            for j in range(len(cluster) - 1):
                mutat_iter = np.random.randint(2, size=10)
                cros_iter = np.random.randint(2, size=10)
                mut_iter = np.random.uniform(0, 1, 10)
                ran = random.choices(cluster, k=5)
                if len(clust_matrix[j][clust_matrix[j] == 1]) > 0:
                    mx = random.choices(np.where(clust_matrix[j] == 1)[0], k=2)
                    ran.append(cluster[mx[0]])
                    ran.append(cluster[mx[1]])
                else:
                    mx = clust_matrix[j].argmax()
                    ran.append(cluster[mx])
                mn = np.argmin(np.where(clust_matrix[j] != 0, clust_matrix[j], 1))
                k = clust_matrix[j][mn]
                ran.append(cluster[mn])
                mn = np.argmin(np.where((clust_matrix[j] != 0) & (clust_matrix[j] != k), clust_matrix[j], 1))
                ran.append(cluster[mn])

                for k, el in enumerate(ran):
                    if cros_iter[k] == 1:
                        new_route = rox(graph, cluster[j], el, n)
                    else:
                        new_route = dpx(graph, cluster[j], el, n)
                    if mut_iter[mutations_counter] <= mutation_prob:
                        if mutat_iter[k] == 1:
                            new_route = mutation_shift(new_route, n)
                        else:
                            new_route = mutation_inversion(new_route, n)
                    route_cluster.append(new_route)
            routes.append(route_cluster)

        population = []

        if len(routes) == 1:
            distances = [length(graph, routes[0][j], n) for j in range(len(routes[0]))]
            num_route = len(clusters[0])
            new_population_cluster, new_population_cluster_distnces = choise(routes[0], distances, n, num_route)
            population.extend(new_population_cluster)
        elif alp == 0:
            nums_route_to_choose = [len(cluster) for cluster in clusters]
            for i in range(len(routes)):
                num_route = nums_route_to_choose[i]
                distances = [length(graph, routes[i][j], n) for j in range(len(routes[i]))]
                new_population_cluster, new_population_cluster_distnces = choise(routes[i], distances, n,
                                                                                 num_route)

                population.extend(new_population_cluster)
        else:
            clust_hunter = np.random.uniform(0, 1, len(routes))
            nums_route_to_choose = [len(cluster) for cluster in clusters]
            routes_sets = []
            for clust in range(len(routes)):
                off_sets = []
                for mas in routes[clust]:
                    off_sets.append(set_format(mas))
                routes_sets.append(off_sets)

            sum_dist = 0
            pop_siz = 0
            balans_alp = 0
            balans_bet = 0
            for i in range(len(routes)):
                if balans:
                    for j in range(len(routes[i])):
                        sum_dist += length(graph, routes[i][j], n)
                pop_siz += len(routes[i])
                if clust_hunter[i] > hunter_prob:
                    balans_bet += 1
                else:
                    balans_alp += 1

            balans_dist = sum_dist/pop_siz
            if alp < 0:
                balans_alp = pop_siz
                balans_bet = pop_siz
            for i in range(len(routes)):
                num_route = nums_route_to_choose[i]
                distances = [length(graph, routes[i][j], n) for j in range(len(routes[i]))]
                if alp < 0:
                    for k in range(len(routes)):
                        if k != i:
                            for j in range(len(routes[i])):
                                if balans:
                                    distances[j] += (alp * balans_dist * clust_dist(routes_sets[i][j],
                                                                                    routes_sets[k])) / balans_alp
                                else:
                                    distances[j] += (alp * clust_dist(routes_sets[i][j],
                                                                      routes_sets[k])) / balans_alp
                elif clust_hunter[i] > hunter_prob:
                    for k in range(len(routes)):
                        if (clust_hunter[k] <= hunter_prob) & (k != i):
                            for j in range(len(routes[i])):
                                if balans:
                                    distances[j] += (bet * balans_dist * clust_dist(routes_sets[i][j],
                                                                                    routes_sets[k])) / balans_alp
                                else:
                                    distances[j] += (bet * clust_dist(routes_sets[i][j],
                                                                      routes_sets[k])) / balans_alp
                else:
                    for k in range(len(routes)):
                        if (clust_hunter[k] > hunter_prob) & (k != i):
                            for j in range(len(routes[i])):
                                if balans:
                                    distances[j] -= (alp * balans_dist * clust_dist(routes_sets[i][j],
                                                                                    routes_sets[k])) / balans_bet
                                else:
                                    distances[j] -= (alp * clust_dist(routes_sets[i][j],
                                                                      routes_sets[k])) / balans_bet
                new_population_cluster, new_population_cluster_distnces = choise(routes[i], distances, n,
                                                                                 num_route)

                population.extend(new_population_cluster)
        if plot:
            population_distnces = [length(graph, population[j], n) for j in range(len(population))]
            print("Poulation size:", len(population), len(population_distnces))
            print(len(population), "min:", min(population_distnces), "avg:", sum(population_distnces) / len(population),
                  "max:", max(population_distnces))
        population_distnces = [length(graph, population[j], n) for j in range(len(population))]
        ans.append(
            [min(population_distnces), sum(population_distnces) / len(population), max(population_distnces), iter])

        if min(population_distnces) == max(population_distnces):
            print(iter)
            print(len(population), "min:", min(population_distnces), "avg:", sum(population_distnces) / len(population),
                  "max:", max(population_distnces))
            print(population[0])
            return ans
    print(max_iter)
    print(population[0])
    print(len(population), "min:", min(population_distnces), "avg:", sum(population_distnces) / len(population),
          "max:", max(population_distnces))
    return ans
