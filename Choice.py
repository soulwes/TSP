import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


def choise(population, distances, n, n_to_choose):
    choosen = []
    choosen_distance = []

    distances_item = [(distances[i], i) for i in range(len(population))]

    distances_item = sorted(distances_item)

    elitism = max(int(0.1 * n_to_choose), 1)
    not_choosen = set([i for i in range(len(population))])
    for i in range(elitism):
        not_choosen.discard(distances_item[i][1])
        choosen_distance.append(distances_item[i][0])
        choosen.append(population[distances_item[i][1]])

    for i in range(n_to_choose - elitism):
        tornament = random.sample(not_choosen, min(max(int(np.sqrt(n_to_choose)), 2), len(not_choosen)))

        value = -1
        current = -1
        for candidate in tornament:
            if distances[candidate] < value or value == -1:
                current = candidate
                value = distances[candidate]

        choosen_distance.append(distances[current])
        choosen.append(population[current])
        not_choosen.discard(current)

    return choosen, choosen_distance
