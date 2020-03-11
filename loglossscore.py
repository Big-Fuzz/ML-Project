import numpy as np
import pandas as pd
from statistics import mean

# Own Libraries/Classes
from setdivision import TestTrain, CrossValidation
from auxilary import score#

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
TrainTestdiv = 20
iterations = 50
k = 11
kernel = "linear"
test_var = 1

knn_array = np.empty(0)
knn_array_2 = np.empty(0)
knnprob_array = np.empty(0)
wknn_array = np.empty(0)
wknnprob_array = np.empty(0)
lr_array = np.empty(0)
svm_array = np.empty(0)


for j in range(iterations):

    trainsets, testsets = CrossValidation(df, TrainTestdiv)
    feat = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                        "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                        "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                        "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
    wfeat = ["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst", "symmetry_worst"]
    #feat = wfeat
    for i in range(len(testsets)):
        train = trainsets[i]
        test = testsets[i]

        knn = score(train, test, feat, "knn", k, kernel)[test_var]
        knn_array = np.append(knn_array, knn)

        #        knn_2 = score(train, test, feat, "knn", k, kernel)[0]
        #       knn_array_2 = np.append(knn_array, knn)

        knnprob = score(train, test, feat, "knnprob", k, kernel)[test_var]
        knnprob_array = np.append(knnprob_array, knnprob)

        wknn = score(train, test, wfeat, "wknn", k, kernel)[test_var]
        wknn_array = np.append(wknn_array, wknn)

        wknnprob = score(train, test, wfeat, "wknnprob", k, kernel)[test_var]
        wknnprob_array = np.append(wknnprob_array, wknnprob)

        lr = score(train, test, feat, "lr", k, kernel)[test_var]
        lr_array = np.append(lr_array, lr)

        svm = score(train, test, feat, "svm", k, kernel)[test_var]
        svm_array = np.append(svm_array, svm)


knn = mean(knn_array)
#knn_2 = mean(knn_array_2)
knnprob = mean(knnprob_array)
wknn = mean(wknn_array)
wknnprob = mean(wknnprob_array)
lr = mean(lr_array)
svm = mean(svm_array)

print("Scores through cross testing after", iterations, "iterations: ")
print("knn score: ", knn)
#print("knn score 2: ", knn_2)
print("knnprob score: ", knnprob)
print("wknn score: ", wknn)
print("wknnprob score: ", wknnprob)
print("lr score: ", lr)
print("svm score: ", svm)
