import numpy as np
from wdistancefunc import weuclidean, weuclidean2

class weightedknn():

    def __init__(self, n, l, v, k, w, sample):
        self.n = np.array(n)
        self.l = np.array(l)
        self.v = np.array(v)
        self.k = k
        self.sample = np.array(sample)
        self.w = np.array(w)

    # Method for vector of nearest neighbours
    def nearestn(self):

        distance = []
        for i in range(self.n.shape[0]):
            D = weuclidean(self.sample, self.n[i],self.w)
            distance.append([D, self.l[i], self.v[i]])

        distance.sort(key=lambda x: x[0])
        y = np.empty([0])

        for i in range(self.k):
            y = np.append(y, (distance[i][2]))

        return y

    # Method for final result (most common nearest neighbour)
    def nverdict(self):

        distance = []

        for i in range(self.n.shape[0]):
            D = weuclidean(self.sample, self.n[i], self.w)
            distance.append([D, self.l[i], self.v[i]])

        distance.sort(key=lambda x: x[0])
        y = np.empty([0])

        for i in range(self.k):
            y = np.append(y, (distance[i][2]))

        a = 0
        b = 0
        s = 0
        r = 0

        for i in range(len(y)):
            if y[i] == y[0]:
                a += 1
            else:
                b += 1
                s = y[i]
        if a > b:
            r = y[0]
        else:
            r = s
        return r

    def hclassprob(self):

        distance = []
        d = np.empty([0])
        probd = np.empty([0])

        for i in range(self.n.shape[0]):
            D = weuclidean(self.sample, self.n[i], self.w)
            distance.append([D, self.l[i], self.v[i]])
            d= np.append(d,D)

        distance.sort(key=lambda x: x[0])
        d.sort()
        y = np.empty([0])

        for i in range(self.k):
            y = np.append(y ,(distance[i][2]))

        for i in range(len(y)):
            if y[i] == 'M':
                y[i] = 1
            else:
                y[i] = 0

        y = y.astype(np.float)

        d = d[:self.k]

        for i in range(len(d)):
            if d[i] == 0:
                probd = [0] * len(d)
                probd[i] = 100
                MP = sum(np.multiply(y, probd))
                return MP

        d = np.asfarray(d, float)
        d = np.reciprocal(d)
        for i in range(len(d)):
            probd = np.append(probd, (100 / sum(d)) * (d[i]))

        MP = sum(np.multiply(y, probd))
        return MP