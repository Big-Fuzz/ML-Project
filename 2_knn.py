from setdivision import TestTrain, TrainValidation, CrossValidation
from knn import knn
from weightedknn import weightedknn

import pandas as pd
import numpy as np
import time
from statistics import mean
np.set_printoptions(threshold=np.inf)


############################################ Uncomment for example Database
data = pd.read_csv("KnnTest.csv")
df = pd.DataFrame(data)

features = df[["X1", "X2", "X3"]]
label_array = df["id"].tolist()
class_array = df["Y"].tolist()
k = 5
ex = features.iloc[4]

k = knn(features, label_array, class_array, k, ex)
y = k.nearestn()
y1 = k.nverdict()
print("The nearest neighbours in the example database are: ", y, "\nThe prediction is: ", y1)



##########################################  Uncomment for Knn test with database

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
features = np.array(df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                        "compactness_mean", "concavity_mean","concave points_mean","symmetry_mean",
                        "fractal_dimension_mean", "radius_se","texture_se", "perimeter_se", "area_se", "smoothness_se",
                        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
                        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
                        "fractal_dimension_worst"]])
label_array = np.array(df["id"].tolist())
class_array = np.array(df["diagnosis"].tolist())
k = 5
s = 12
sample = features[s]
o = knn(features, label_array, class_array, k, sample)
y = o.nearestn()
y1 = o.nverdict()

print("The nearest neighbours: ", y, "\nThe prediction: ", y1, "\nThe true value: ", class_array[s])



############################################## Uncomment for Finding Score

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
train, test = TestTrain(df, 20)

train_features = np.array(train[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                                 "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
                                 "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
                                 "smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
                                 "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst",
                                 "smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
                                 "symmetry_worst","fractal_dimension_worst"]])
train_labels = np.array(train["id"].tolist())
train_classes = np.array(train["diagnosis"].tolist())

test_features = np.array(test[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                               "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
                               "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
                               "smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
                               "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst",
                               "smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
                               "symmetry_worst","fractal_dimension_worst"]])
test_labels = np.array(test["id"].tolist())
test_classes = np.array(test["diagnosis"].tolist())

k = 10
correct = 0
incorrect = 0
for s in range(len(test_features)):
    sample = test_features[s]
    o = knn(train_features, train_labels, train_classes, k, sample)
    y = o.nearestn()
    y1 = o.nverdict()

    if y1 == test_classes[s]:
        correct += 1
    else:
        incorrect += 1

accuracy = (correct /(correct + incorrect)) *100

print("The accuracy of one iteration through a Train-Test split in the Wisconsin database is: ", accuracy)



############################################ Uncomment for finding score averaged through iterations

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
k = 20
accuracy_list = []
time_list = []
iterations = 1

for i in range(iterations):
    train, test = TestTrain(df, 20)

    train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                            "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                            "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                            "symmetry_worst", "fractal_dimension_worst"]])
    train_labels = np.array(train["id"])
    train_classes = np.array(train["diagnosis"])

    test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                          "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                          "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                          "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                          "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                          "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                          "symmetry_worst", "fractal_dimension_worst"]])
    test_labels = np.array(test["id"])
    test_classes = np.array(test["diagnosis"])

    correct = 0
    incorrect = 0

    for s in range(len(test_features)):
        sample = test_features[s]
        o = knn(train_features, train_labels, train_classes, k, sample)
        y = o.nverdict()

        if y == test_classes[s]:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct /(correct + incorrect)) *100

    accuracy_list.append(accuracy)


accuracy_average = mean(accuracy_list)

print("The accuracy of", iterations, "iterations through a Train-Test split in the Wisconsin database is: ",
      accuracy_average)



########################################### Uncomment for finding average score with Crossfold Testing
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
trains, tests = CrossValidation(df, 20)  # Dividing Dataframes into train and test set
k = 20
accuracy_list = []


# Looping through each train and test set together
for i in range(len(trains)):
    train = trains[i]
    test = tests[i]

    # Specifiying Features
    train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                            "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                            "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                            "symmetry_worst", "fractal_dimension_worst"]])
    test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                          "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                          "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                          "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                          "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                          "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                          "symmetry_worst", "fractal_dimension_worst"]])

    # Specifying Labels
    train_labels = np.array(train["id"])
    test_labels = np.array(test["id"])

    # Specifying Classes
    train_class = np.array(train["diagnosis"])
    test_class = np.array(test["diagnosis"])

    correct = 0
    incorrect = 0

    for s in range(len(test_class)):
        sample = test_features[s]
        o = knn(train_features, train_labels, train_class, k, sample)
        y = o.nverdict()

        if y == test_class[s]:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct / (correct + incorrect)) * 100
    e2 = time.time()
    accuracy_list.append(accuracy)

accuracy_average = mean(accuracy_list)
print("The accuracy of one iteration through a Cross-Testing split in the Wisconsin database is: ",
      accuracy_average)



################################################## Uncomment for nearest distance probabilities
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
train, test = TestTrain(df, 20)

train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                        "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                        "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                        "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                        "symmetry_worst", "fractal_dimension_worst"]])
train_labels = np.array(train["id"])
train_classes = np.array(train["diagnosis"])

test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                      "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                      "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                      "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                      "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                      "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                      "symmetry_worst", "fractal_dimension_worst"]])
test_labels = np.array(test["id"])
test_classes = np.array(test["diagnosis"])

k = 10
s = 5
sample = test_features[s]
o = knn(train_features, train_labels, train_classes, k, sample)
d, p = o.nearestdistanceprob()
l = o.nearestn()
n = o.nverdict()

print("The following are results of using the nearestdistanceprob method of the knn class:")
print("The nearest neighbours: ", l)
print("The prediction: ", n)
print("The true value: ", test_classes[s])
print("The distance vectors",d)
print("The probabilities: ", p)



########################################### Uncomment for finding score average with weighted knn
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
k = 10
accuracy_list = []
iterations = 1

for i in range(iterations):

    train, test = TestTrain(df, 20)

    train_features = train[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                            "symmetry_worst"]]
    train_labels = train["id"].tolist()
    train_classes = train["diagnosis"].tolist()

    test_features = np.array(test[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                          "symmetry_worst"]])
    test_labels = np.array(test["id"])
    test_classes = np.array(test["diagnosis"])

    totcor = np.array([0.456902821396798, 0.782914137173759, 0.421464861066403, 0.79356601714127,
                       0.416294311048619])
    w = np.empty([0])
    for i in range(len(totcor)):
        w = np.append(w, (totcor[i] / np.sum(totcor)))

    correct = 0
    incorrect = 0

    for s in range(len(test_features)):
        sample = test_features[s]
        o = weightedknn(train_features, train_labels, train_classes, k, w, sample)
        y = o.nverdict()
        if y == test_classes[s]:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct / (correct + incorrect)) * 100
    accuracy_list.append(accuracy)

accuracy_average = mean(accuracy_list)
print("The average accuracy after", iterations, "iterations using weightedknn after manually assigning weights ",
      accuracy_average)



############################################### Uncomment for finding score average with weighted knn EXTENDED to
#                                               (autotamatic extraction of features from correlation matrix)
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
k = 10
accuracy_list = []
iterations = 1

for i in range(iterations):
    train, test = TestTrain(df, 20)

    train_features = np.array(train[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                                     "symmetry_worst"]])
    train_labels = np.array(train["id"])
    train_classes = np.array(train["diagnosis"])

    test_features = np.array(test[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                                   "symmetry_worst"]])
    test_labels = np.array(test["id"])
    test_classes = np.array(test["diagnosis"])

    dia = train_classes
    dia = np.array(dia)
    dianp = np.empty(0)

    for i in range(len(dia)):
        if dia[i] == "B":
            dianp = np.append(dianp, np.float64(0.0))
        else:
            dianp = np.append(dianp, np.float64(1.0))

    df1 = train[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                "symmetry_worst"]].assign(dia=dianp)
    corr = df1.corr()
    w = corr[["dia"]]
    w = w.drop(index='dia')
    totcor = np.array(w)
    w = np.empty([0])
    for i in range(len(totcor)):
        w = np.append(w, (totcor[i] / np.sum(totcor)))

    correct = 0
    incorrect = 0

    for s in range(len(test_features)):
        sample = test_features[s]
        o = weightedknn(train_features, train_labels, train_classes, k, w, sample)
        y = o.nverdict()
        if y == test_classes[s]:
            correct += 1
        else:
            incorrect += 1
    accuracy = (correct /(correct + incorrect)) *100
    accuracy_list.append(accuracy)

accuracy_average = mean(accuracy_list)

print("The accuracy of", iterations, "iterations with weightedknn by taking the initial weights from the correlation "
                                     "matrix of the database of the Wisconsin database is: ", accuracy_average)



################################################### Uncomment finding average score weightedknn with the higher
#                                                   probability
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
k = 10
accuracy_list = np.empty(0)
iterations = 1

for i in range(iterations):

    train, test = TestTrain(df, 20)

    train_features = np.array(train[
        ["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst", "symmetry_worst"]])
    train_labels = np.array(train["id"])
    train_classes = np.array(train["diagnosis"])

    test_features = np.array(test[
        ["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst", "symmetry_worst"]])
    test_features = np.array(test_features)
    test_labels = np.array(test["id"])
    test_classes = np.array(test["diagnosis"])

    dia = train_classes
    dia = np.array(dia)
    dianp = np.empty(0)

    for i in range(len(dia)):
        if dia[i] == "B":
            dianp = np.append(dianp, np.float64(0.0))
        else:
            dianp = np.append(dianp, np.float64(1.0))

    df1 = train[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
                 "symmetry_worst"]].assign(dia=dianp)
    corr = df1.corr()
    w = corr[["dia"]]
    w = w.drop(index='dia')
    totcor = np.array(w)
    w = np.empty([0])
    for i in range(len(totcor)):
        w = np.append(w, (totcor[i] / np.sum(totcor)))

    correct = 0
    incorrect = 0

    for s in range(len(test_features)):

        sample = test_features[s]

        o = weightedknn(train_features, train_labels, train_classes, k, w, sample)
        y = o.hclassprob()

        if y >= 50:
            y = "M"
        else:
            y = "B"

        if y == test_classes[s]:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct / (correct + incorrect)) * 100
    accuracy_list = np.append(accuracy_list, accuracy)


accuracy_average = mean(accuracy_list)
print(accuracy_average)
