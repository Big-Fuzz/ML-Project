import numpy as np
from distance import dist

""" The three Steps of Knn:
        1.) Normalize the data
        2.) Find the k nearest neighbors
        3.) Classify the new point based on those neighbors """


class knn():

    def __init__(self, features, labels, classes, k, sample):
        self.n = np.array(features)
        self.l = np.array(labels)
        self.v = np.array(classes)
        self.k = k
        self.sample = np.array(sample)




    # Method for vector of nearest neighbours
    def nearestn(self):

        distance = []

        for i in range(self.n.shape[0]):
            D = dist(self.sample, self.n[i])
            distance.append([D.euclidean(), self.l[i], self.v[i]])

        distance.sort(key=lambda x: x[0])
        y = np.empty([0])

        for i in range(self.k):
            y = np.append(y, (distance[i][2]))

        return y

    # Method for final result (most common nearest neighbour)
    def nverdict(self):

        distance = []

        for i in range(self.n.shape[0]):
            D = dist(self.sample, self.n[i])
            distance.append([D.euclidean(), self.l[i], self.v[i]])

        distance.sort(key=lambda x: x[0])

        y = np.empty([0])

        for i in range(self.k):
            y = np.append(y, (distance[i][2]))

        a = 0
        b = 0
        s = 0

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




    # Method that returns the distances of the nearest neighbours along with the vector of them
    def nearestdistance(self):

        distance = []
        d = np.empty([0])

        for i in range(self.n.shape[0]):
            D = dist(self.sample, self.n[i])
            distance.append([D.euclidean(), self.l[i], self.v[i]])
            d= np.append(d,D.euclidean())

        distance.sort(key=lambda x: x[0])
        d.sort()

        distance = distance[:self.k]
        d = d[:self.k]

        return distance, d




    # Method that returns the probabilities and the distances
    def nearestdistanceprob(self):

        distance = []
        d = np.empty([0])
        probd = np.empty([0])

        for i in range(self.n.shape[0]):
            D = dist(self.sample, self.n[i])
            distance.append([D.euclidean(), self.l[i], self.v[i]])
            d= np.append(d,D.euclidean())

        distance.sort(key=lambda x: x[0])
        d.sort()

        distance = distance[:self.k]
        d = d[:self.k]

        for i in range(len(d)):
            if d[i] == 0:
                probd = [0] * len(d)
                probd[i] = 100
                return distance, probd

        d = np.asfarray(d, float)
        d = np.reciprocal(d)
        for i in range(len(d)):
            probd = np.append(probd, (100 / sum(d)) * (d[i]))

        return distance, probd




    # Method that returns the probability of the higher class
    def hclassprob(self):

        distance = []
        d = np.empty([0])
        probd = np.empty([0])

        for i in range(self.n.shape[0]):
            D = dist(self.sample, self.n[i])
            distance.append([D.euclidean(), self.l[i], self.v[i]])
            d= np.append(d,D.euclidean())

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