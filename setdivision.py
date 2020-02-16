import pandas as pd
from random import seed
from random import randint
import time


def TestTrain(df, testpercentage):

    Test = pd.DataFrame()
    randlist = []
    for i in range(int((testpercentage/ 100) * df.shape[0])):
        seed(time.time())
        value = randint(0, df.shape[0]-1)
        if value in randlist:
            while value in randlist:
                value = randint(0, df.shape[0] - 1)
        randlist.append(value)
    for i in range(len(randlist)):
            Test = Test.append(df[randlist[i]:randlist[i]+1])
    Train = df.drop(randlist, axis=0)
    return (Train, Test)


def TrainValidation(df, validationpercentage):

    Validation = pd.DataFrame()
    randlist = []
    for i in range(int((validationpercentage/ 100) * df.shape[0])):
        seed(time.time())
        value = randint(0, df.shape[0]-1)
        if value in randlist:
            while value in randlist:
                value = randint(0, df.shape[0] - 1)
        randlist.append(value)
    for i in range(len(randlist)):
        # if i < 1:
        #     Validation = df[randlist[i]:randlist[i]+1]
        # else:
            Validation = Validation.append(df[randlist[i]:randlist[i]+1])
    Train = df.reset_index()
    Train = Train.drop('index', 1)
    Train = Train.drop(randlist, axis=0)
    return (Train, Validation)

def CrossValidation(df, percentage):
    randlist = []
    Test = {}
    Train = {}
    df = df.reset_index()
    df = df.drop('index', 1)
    for i in range((df.shape[0])):
        seed(time.time())
        value = randint(0, df.shape[0]-1)
        if value in randlist:
            while value in randlist:
                value = randint(0, df.shape[0] - 1)
        randlist.append(value)


    for i in range(int(100/percentage)):

        test_i = pd.DataFrame()

        train_i = pd.DataFrame()
        for j in range(int((percentage/100) * len(randlist))):

            test_i = test_i.append(df[randlist[j + i * int(percentage/100 * len(randlist))]:randlist[j + i * int(percentage/100 * len(randlist))] + 1])
        Test[i] = test_i

        r = randlist[i * int(percentage/100 * len(randlist)) : i * int(percentage/100 * len(randlist)) + (int(percentage/100 * len(randlist))) ]

        train_i = df.drop(r, axis=0)

        Train[i] = train_i

    return (Train, Test)