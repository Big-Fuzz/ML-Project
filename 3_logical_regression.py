import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import time
from statistics import mean

from setdivision import TestTrain, TrainValidation, CrossValidation

################################ UNCOMMENT FOR TESTING FOR ONE SAMPLE OF THE WISCONSIN CANCER DATABASE

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

train, test = TestTrain(df, 20)  # Dividing Dataframes into train and test set

# Specifiying Features
train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                                 "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                                 "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                                 "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                                 "area_worst","smoothness_worst", "compactness_worst", "concavity_worst",
                                 "concave points_worst","symmetry_worst", "fractal_dimension_worst"]])

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

# Normalization of Dataframes
train_features = preprocessing.scale(train_features)
test_features = preprocessing.scale(test_features)

# Implementing and fitting model
model = LogisticRegression()
model.fit(train_features, train_class)
test_features = np.array(test_features)
s = 5
sample = test_features[s]
sample = sample.reshape(-1, 1)
sample = sample.transpose()
predictions = model.predict(sample)

# Print and compare true value with prediction
print("The true value:", test_class[s])
print("The predicted value: ", predictions[0])

##################################################### UNCOMMENT FOR FINDING SCORE
# Looping to find accuracy/test error and score

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

# Dividing Dataframes into train and test set
train, test = TestTrain(df, 20)

# Specifiying Features
train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                 "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                                 "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                                 "smoothness_se","compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                                 "fractal_dimension_se","radius_worst", "texture_worst", "perimeter_worst",
                                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                 "concave points_worst", "symmetry_worst","fractal_dimension_worst"]])

test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                               "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                               "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                               "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                               "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                               "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                               "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]])

# Specifying Classes
train_class = np.array(train["diagnosis"])
test_class = np.array(test["diagnosis"])

# Normalization of Dataframes
train_features = preprocessing.scale(train_features)
test_features = preprocessing.scale(test_features)

# Implementing and fitting model
model = LogisticRegression()
model.fit(train_features, train_class)
correct = 0
incorrect = 0

for i in range(len(test_class)):
    s = i
    sample = test_features[s]
    sample = sample.reshape(-1,1)
    sample = sample.transpose()

    predictions= model.predict(sample)

    if test_class[s] == predictions:
        correct += 1
    else:
       incorrect += 1

accuracy = (correct /(correct + incorrect)) *100
print("The accuracy of one iteration through a Train-Test split in the Wisconsin database is: ", accuracy)



#############################################   UNCOMMENT FOR FINDING SCORE WITH CROSSFOLD TESTING

data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

# Dividing Dataframes into train and test set
trains, tests = CrossValidation(df, 20)
accuracy_list = []

for i in range(len(trains)):  # Looping through each train and test set together
    train = trains[i]
    test = tests[i]

    # Specifiying Features
    train_features = np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                     "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                                     "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                                     "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                                     "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                                     "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
                                     "concavity_worst", "concave points_worst", "symmetry_worst",
                                     "fractal_dimension_worst"]])

    test_features = np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                                   "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                                   "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                                   "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
                                   "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                                   "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
                                   "concavity_worst", "concave points_worst", "symmetry_worst",
                                   "fractal_dimension_worst"]])

    # Specifying Classes
    train_class = np.array(train["diagnosis"])
    test_class = np.array(test["diagnosis"])

    # Normalization of Dataframes
    train_features = preprocessing.scale(train_features)
    test_features = preprocessing.scale(test_features)

    # Implementing and fitting model
    model = LogisticRegression()
    model.fit(train_features, train_class)
    correct = 0
    incorrect = 0

    # Finding score by calculating predictions for each sample (predictions) in the test set and then comparing
    # it with the true value (test_class[s]) .
    for j in range(len(test_class)):
        s = i
        sample = test_features[s]
        sample = sample.reshape(-1,1)
        sample = sample.transpose()
        predictions= model.predict(sample)

        if test_class[s] == predictions:
            correct += 1
        else:
            incorrect += 1

    accuracy = (correct /(correct + incorrect)) *100
    accuracy_list.append(accuracy)

accuracy_average = mean(accuracy_list)
print("The accuracy of one iteration through a Train-Validation split in the Wisconsin database is: ", accuracy_average)



################################################### UNCOMMENT FOR FINDING PROBABILITY
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

# Normalization of Dataframes
train_features = preprocessing.scale(train_features)
test_features = preprocessing.scale(test_features)

# Implementing and fitting model
model = LogisticRegression()
model.fit(train_features, train_class)

s = 3
sample = test_features[s]
sample = sample.reshape(-1, 1)
sample = sample.transpose()
predictions = model.predict(sample)

# Print and compare true value with prediction
print("True value:", test_class[s])
print("Predicted value: ", predictions[0])
print("Probability of predicted value being M",model.predict_proba(sample)[:,1])



################################################# UNCOMMENT FOR ACCURACY WITH CROSSVALIDATION
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
traintestdiv = 20
crossvalidationdiv = 20
iterations = 1000
final_accuracy_list = []

# Dividing Dataframes into train, validation, and test set
trainset, test = TestTrain(df, traintestdiv)

test_class = np.array(test["diagnosis"])
test_features = np.array(test[
    ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
     "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
     "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
     "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
     "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]])

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

        # Normalization of Dataframes
        train_features = preprocessing.scale(train_features)
        validation_features = preprocessing.scale(validation_features)

        # Implementing and fitting model
        model = LogisticRegression()
        model.fit(train_features, train_class)
        correct = 0
        incorrect = 0

        # Finding score by calculating predictions for each sample (predictions) in the test set and then
        # comparing it with the true value (test_class[s]).
        for z in range(len(validation_class)):
            s = z
            sample = validation_features[s]
            sample = sample.reshape(-1,1)
            sample = sample.transpose()
            predictions= model.predict(sample)

            if validation_class[s] == predictions:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct / len(validation_class)) *100
        accuracy_list.append(accuracy)

    accuracy_average = mean(accuracy_list)
    final_accuracy_list.append(accuracy_average)

final_accuracy_average = mean(final_accuracy_list)
print("The average accuracy after", iterations, "iterations using Logistic Regression on the Wisconsin database using cross validation", final_accuracy_average)