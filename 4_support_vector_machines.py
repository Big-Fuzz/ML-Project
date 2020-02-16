from sklearn import svm
import pandas as pd
import numpy as np
from statistics import mean
from setdivision import TestTrain, TrainValidation, CrossValidation



######################################## Testing basic functions of SkLearn SVM (Support Vector Machine)
X = [[0, 0], [2, 5], [8, 6]]
y = [0, 1, 0]
clf = svm.SVC( C=1, kernel = 'sigmoid', probability= True)
clf.fit(X, y)

p = clf.predict([[5, 5]])
proba = clf.predict_proba([[5, 5]])
print(proba)
print("Prediction of p: ",p)
print("Support Vectors: ",clf.support_vectors_)
print("Indices of support vectors: ",clf.support_)
print("Number of support vectors for each class: ",clf.n_support_)
print("Parameters: ", clf.get_params())



########################################## Testing SVM for Breast Cancer Detection Database
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
train, test = TestTrain(df, 20)  # Dividing Dataframes into train and test set

# Specifiying Features
train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                 "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                                 "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                                 "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                                 "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                 "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]])

test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                               "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                               "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                               "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                               "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                               "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                               "symmetry_worst", "fractal_dimension_worst"]])

# Specifying Classes
train_class = np.array(train["diagnosis"])
test_class = np.array(test["diagnosis"])

# Implementing and fitting model
model = svm.SVC(kernel = 'linear', C=1, probability=True)
model.fit(train_features, train_class)
test_features = np.array(test_features)
s = 67
sample = test_features[s]
sample = sample.reshape(-1, 1)
sample = sample.transpose()
predictions = model.predict(sample)
proba = model.predict_proba(sample)
# Print and compare true value with prediction
print("True value:", test_class[s])
print("Predicted value: ", predictions[0])
print("Probability: ", proba[:, 1][0])



############################################### UNCOMMENT FOR FINAL ACCURACY
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
traintestdiv = 20
crossvalidationdiv = 20
iterations = 10
final_accuracy_list = []
kernel = "linear"
trainset, test = TestTrain(df, traintestdiv) # Dividing Dataframes into train, validation, and test set
test_class = test["diagnosis"]
test_class = np.array(test_class)
test_features = test[
    ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
     "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
     "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
     "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
     "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]]

for j in range(iterations):
    trains, validations = CrossValidation(trainset, crossvalidationdiv)
    accuracy_list = []

    for i in range(len(trains)):
        train = trains[i]
        validation = validations[i]

        # Specifiying Features
        train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                                         "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                                         "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
                                         "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                                         "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst",
                                         "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                                         "compactness_worst", "concavity_worst", "concave points_worst",
                                         "symmetry_worst", "fractal_dimension_worst"]])

        validation_features = np.array(validation[["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                                                   "smoothness_mean", "compactness_mean", "concavity_mean",
                                                   "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
                                                   "radius_se", "texture_se", "perimeter_se", "area_se",
                                                   "smoothness_se", "compactness_se", "concavity_se",
                                                   "concave points_se", "symmetry_se", "fractal_dimension_se",
                                                   "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
                                                   "smoothness_worst", "compactness_worst", "concavity_worst",
                                                   "concave points_worst", "symmetry_worst",
                                                   "fractal_dimension_worst"]])

        # Specifying Classes
        train_class = np.array(train["diagnosis"])
        validation_class = np.array(validation["diagnosis"])

        # Implementing and fitting model
        model = svm.SVC(C=1, kernel=kernel, degree=3)
        model.fit(train_features, train_class)
        correct = 0
        incorrect = 0

        # Finding score by calculating predictions for each sample (predictions) in the test set and then comparing
        # it with the true value (test_class[s]).
        for z in range(len(validation_class)):
            s = z
            sample = validation_features[s]
            sample = sample.reshape(-1, 1)
            sample = sample.transpose()
            predictions = model.predict(sample)

            if validation_class[s] == predictions:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct / (len(validation_class))) * 100

        final_accuracy_list.append(accuracy)

final_accuracy_average = mean(final_accuracy_list)
print(final_accuracy_average)