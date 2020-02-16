import numpy as np
import pandas as pd
from statistics import mean

# Own Libraries/Classes
from setdivision import TestTrain, CrossValidation
from auxilary import score

""""
############################################## Description ############################################################

score(train, test, "method", wtrain, wtest, k, "kernel"):

train: Training set

test: Test set

method:
1.) KNN Score                                             knn
2.) KNN Score with Probability                            knnprob
3.) Weighted KNN Score                                    wknn
4.) Weighted KNN Score with probability                   wknnprob
5.) LR Score                                              lr
6.) SVM Score                                             svm

wtrain: training features for weighted methods

wtest: test features for weighedted methods

k : number of nearest neighbours

kernel: kernel for the svm method:

1.) linear
2.) poly
3.) rbf
4.) sigmoid
"""



################################################## Finding the best K for the Knn method
# data = pd.read_csv("data.csv")
# df = pd.DataFrame(data)
# TrainTestdiv = 20
# iterations = 50
# knn_array = np.empty(0)
# knn_matrix = np.empty(0)
# neighbours = 20
# kernel = "poly"
#
# # Iterate to get more than one score for each K
# for j in range(iterations):
#     knn_array = np.empty(0)
#     train, test = TestTrain(df, TrainTestdiv)
#     wtrain = 0
#     wtest = 0
#
#     # loop through neighbours finding score with each k
#     for k in range(1,neighbours):
#         knn = score(train, test, "knn", wtrain, wtest, k, kernel)
#         knn_array = np.append(knn_array, knn)
#     if j == 0:
#         knn_matrix = knn_array # Initialize first row of matirix
#     else:
#         knn_matrix = np.vstack((knn_matrix, knn_array)) # stack rows into matrix
# print(knn_matrix)
#
# final = np.empty(0)
# for i in range(len(knn_array)):
#     final = np.append(final, mean(knn_matrix[:,i]))
# print(final)
# print(max(final))
# k =  np.where(final == max(final))
# print(k)
#
#
#
# ################################################ Finding the best kernal for the svm method
# data = pd.read_csv("data.csv")
# df = pd.DataFrame(data)
# TrainTestdiv = 20
# iterations = 20
# svm_array = np.empty(0)
# svm_matrix = np.empty(0)
# kernels= np.array(["linear","poly","rbf","sigmoid"])
# k = 10
# # Iterate to get more than one score for each K
# for j in range(iterations):
#     svm_array = np.empty(0)
#     train, test = TestTrain(df, TrainTestdiv)
#     wtrain = 0
#     wtest = 0
#
#     # loop through neighbours finding score with each k
#     for f in range(len(kernels)):
#         svm = score(train, test, "svm", wtrain, wtest, k, kernels[f])
#         svm_array = np.append(svm_array, svm)
#     if j == 0:
#         svm_matrix = svm_array # Initialize first row of matirix
#     else:
#         svm_matrix = np.vstack((svm_matrix, svm_array)) # stack rows into matrix
# print(svm_matrix)
#
# final = np.empty(0)
# for i in range(len(svm_array)):
#     final = np.append(final, mean(svm_matrix[:,i]))
# print(final)
# print(max(final))
# k = np.where(final == max(final))
# print(kernels[k])



################################################# Finding scores of all methods
# data = pd.read_csv("data.csv")
# df = pd.DataFrame(data)
# TrainTestdiv = 20
# iterations = 3
# k = 10
# kernel = "linear"
#
# knn_array = np.empty(0)
# knnprob_array = np.empty(0)
# knnpen_array = np.empty(0)
# wknn_array = np.empty(0)
# wknnprob_array = np.empty(0)
# wknnpen_array = np.empty(0)
# lr_array = np.empty(0)
# lrpen_array = np.empty(0)
# svm_array = np.empty(0)
# svmpen_array = np.empty(0)
#
# for j in range(iterations):
#     train, test = TestTrain(df, TrainTestdiv)
#     wtrain = train[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
#                     "symmetry_worst"]]
#     wtest = test[["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst",
#                   "symmetry_worst"]]
#
#     # wtrain = train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
#     #                          "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
#     #                          "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
#     #                          "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
#     #                          "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
#     #                          "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
#     #                          "symmetry_worst", "fractal_dimension_worst"]]
#     #
#     # wtest = test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
#     #               "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
#     #               "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
#     #               "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
#     #               "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
#     #               "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]
#
#     knn = score(train, test, "knn", wtrain, wtest, k, kernel)[0]
#     knn_array = np.append(knn_array, knn)
#
#     knnprob = score(train, test, "knnprob", wtrain, wtest, k, kernel)[0]
#     knnprob_array = np.append(knnprob_array, knnprob)
#
#     wknn = score(train, test, "wknn", wtrain, wtest, k, kernel)[0]
#     wknn_array = np.append(wknn_array, wknn)
#
#     wknnprob = score(train, test, "wknnprob", wtrain, wtest, k, kernel)[0]
#     wknnprob_array = np.append(wknnprob_array, wknnprob)
#
#     lr = score(train, test, "lr", wtrain, wtest, k, kernel)[0]
#     lr_array = np.append(lr_array, lr)
#
#     svm = score(train, test, "svm", wtrain, wtest, k, kernel)[0]
#     svm_array = np.append(svm_array, svm)
#
#
# knn = mean(knn_array)
# knnprob = mean(knnprob_array)
# wknn = mean(wknn_array)
# wknnprob = mean(wknnprob_array)
# lr = mean(lr_array)
# svm = mean(svm_array)
#
# print("Average scores after", iterations, "iterations: ")
# print("knn score: ", knn)
# print("knnprob score: ", knnprob)
# print("wknn score: ", wknn)
# print("wknnprob score: ", wknnprob)
# print("lr score: ", lr)
# print("svm score: ", svm)



####################################### Finding scores of all methods through Cross-Testing
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
TrainTestdiv = 20
iterations = 10 #*5
k = 10
kernel = "linear"

knn_array = np.empty(0)
knnprob_array = np.empty(0)
knnpen_array = np.empty(0)
wknn_array = np.empty(0)
wknnprob_array = np.empty(0)
wknnpen_array = np.empty(0)
lr_array = np.empty(0)
lrpen_array = np.empty(0)
svm_array = np.empty(0)
svmpen_array = np.empty(0)

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
    feat = wfeat
    for i in range(len(testsets)):
        train = trainsets[i]
        test = testsets[i]
        #wtrain = train[wfeat]# train[
                #["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst", "symmetry_worst"]]
        #wtest = test[
                #["texture_worst", "perimeter_worst", "smoothness_worst", "concave points_worst", "symmetry_worst"]]
        #train = np.array(wtrain)
        #wtrain = train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        #                "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
        #                "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        #                "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
        #                "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
        #                "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
        #                "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]

        #wtest = test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        #              "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
        #              "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        #              "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
        #              "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        #              "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
        #              "symmetry_worst", "fractal_dimension_worst"]]

        knn = score(train, test, feat, "knn", k, kernel)[0]
        knn_array = np.append(knn_array, knn)

        knnprob = score(train, test, feat, "knnprob", k, kernel)[0]
        knnprob_array = np.append(knnprob_array, knnprob)

        wknn = score(train, test, wfeat, "wknn", k, kernel)[0]
        wknn_array = np.append(wknn_array, wknn)

        wknnprob = score(train, test, wfeat, "wknnprob", k, kernel)[0]
        wknnprob_array = np.append(wknnprob_array, wknnprob)

        lr = score(train, test, feat, "lr", k, kernel)[0]
        lr_array = np.append(lr_array, lr)

        svm = score(train, test, feat, "svm", k, kernel)[0]
        svm_array = np.append(svm_array, svm)


knn = mean(knn_array)
knnprob = mean(knnprob_array)
wknn = mean(wknn_array)
wknnprob = mean(wknnprob_array)
lr = mean(lr_array)
svm = mean(svm_array)

print("Scores through cross testing after", iterations, "iterations: ")
print("knn score: ", knn)
print("knnprob score: ", knnprob)
print("wknn score: ", wknn)
print("wknnprob score: ", wknnprob)
print("lr score: ", lr)
print("svm score: ", svm)