import numpy as np
import pandas as pd
from statistics import mean

# Own Libraries/Classes
from setdivision import TestTrain
from auxilary import joint_score, find_weights_joint


data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
crossvalidationdiv = 20
k = 11
#train, test = TestTrain(df, 20)
features = np.array(
    ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
     "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
     "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
     "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
     "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"])
classes = np.array(["diagnosis"])
labels = np.array(["id"])
k_fold_iterations = 3
method1 = "knn"
method2 = "lr"
i = 0
w = np.array([0.5, 0.5])
stepratio = 0.1
direction_limit = 3
step_limit = 10
testvairable = 3
train_test_gap_list = np.empty(0)
test_score_list = np.empty(0)
wlr_list = np.empty(0)
wknn_list = np.empty(0)
repititions = 5

for t in range(0, repititions):
    train, test = TestTrain(df, 20)
    y = find_weights_joint(train, features, classes, labels, crossvalidationdiv, k_fold_iterations, i, w, stepratio, k,
                           method1, method2, direction_limit, step_limit, testvairable)

    training_score = y[0][0]
    w_lr = y[0][1][0]
    w_knn = y[0][1][1]
    w_new = np.array([w_lr, w_knn])
    test_score = joint_score(train, test, features, classes, labels, k, w_new, method1, method2)[testvairable]
    train_test_gap = abs(training_score - test_score)

    print("Training score:", training_score)
    print("Test score:", test_score)
    print("Train test gap:", train_test_gap)

    train_test_gap_list = np.append(train_test_gap_list, train_test_gap)
    wlr_list = np.append(wlr_list, w_lr)
    wknn_list = np.append(wknn_list, w_knn)
    test_score_list = np.append(test_score_list, test_score)

average_test_score = mean(test_score_list)
average_train_test_gap = mean(train_test_gap_list)
average_lr_weight = mean(wlr_list)
average_knn_weight = mean(wknn_list)

print("Average test score: ", average_test_score)
print("Average train-test gap: ", average_train_test_gap)
print("Average weight for logistic regression: ", average_lr_weight)
print("Average weight for Knn: ", average_knn_weight)