import numpy as np


def weuclidean(v1, v2, w):

    v1 = np.array(v1)
    v2 = np.array(v2)
    w = np.array(w)
    if len(v1) != len(v2):
        return print(" The lists are not of equal lengths")
    else:
        d = 0
        for i in range(len(v1)):
            if w[i] == 0:
                w[i] = 0.000001
            d += ((v1[i] - v2[i])/w[i]) ** 2
        return d ** 0.5


def weuclidean2(v1, v2, w):

    v1 = np.array(v1)
    v2 = np.array(v2)
    w = np.array(w)
    if len(v1) != len(v2):
        return print(" The lists are not of equal lengths")
    else:
        d = 0
        for i in range(len(v1)):
            d += ((v1[i] - v2[i]) ** 2)/w[i]
        return d ** 0.5