import numpy as np

class dist():

    def __init__(self, list_one, list_two):
        self.list_one = np.array(list_one)
        self.list_two = np.array(list_two)

# Euclidean Distance
    def euclidean(self):
        self.list_one = np.array(self.list_one)
        self.list_two = np.array(self.list_two)
        if len(self.list_one) != len(self.list_two):
            return print(" The lists are not of equal lengths")
        else:
            d = 0
            for i in range(len(self.list_one)):
                d += (self.list_one[i] - self.list_two[i]) ** 2
            return d ** 0.5

# Manhattan Distance
    def  manhattan(self):
        if len(self.list_one) != len(self.list_two):
            return print(" The lists are not of equal lengths")
        else:
            d = 0
            for i in range(len(self.list_one)):
                d += abs(self.list_one[i] - self.list_two[i])
            return d

# Hamming Distance
    def  hamming(self):
        if len(self.list_one) != len(self.list_two):
            return print(" The lists are not of equal lengths")
        else:
            d = 0
            for i in range(len(self.list_one)):
                if (self.list_one[i]) != (self.list_two[i]):
                    d += 1
            return d

