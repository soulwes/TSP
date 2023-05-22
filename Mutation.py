import numpy as np
import warnings
warnings.filterwarnings("ignore")


def mutation_shift(item, n):
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)

    if a > b:
        a, b = b, a

    if a == b and b == 0:
        b += 1
    if a == b and b == n:
        a -= 1

    c = np.random.randint(0, n - (b - a))

    to_insert = item[a:b]
    mutated = []

    for i in range(n - (b - a)):
        counter = i
        if i >= a:
            counter = i + ((b - a))
        if i == c:
            for j in to_insert:
                mutated.append(j)
        mutated.append(item[counter])

    if len(mutated) != len(item):
        for j in to_insert:
            mutated.append(j)

    return mutated


def mutation_inversion(item, n):
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)

    if a > b:
        a, b = b, a
    if a == b and b == 0:
        b += 1
    if a == b and b == n:
        a -= 1

    to_insert = item[a:b]
    to_insert = np.flip(to_insert)

    mutated = [to_insert[i - a] if a <= i and i < b else item[i] for i in range(n)]

    return mutated