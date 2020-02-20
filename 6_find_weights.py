import numpy as np
import pandas as pd
from statistics import mean

# Own Libraries/Classes
from setdivision import TestTrain, TrainValidation, CrossValidation
from auxilary import score, joint_score_prev_current_next,  joint_score, find_weights_joint



data = pd.read_csv("data.csv")
df = pd.DataFrame(data)
crossvalidationdiv = 20
k = 10
training,testing = TestTrain(df, 20)
features = np.array(
    ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
     "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
     "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
     "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
     "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"])

classes = np.array(["diagnosis"])
labels = np.array(["id"])
k_fold_iterations = 10
method1 = "knn"
method2 = "lr"
i = 0 # 0 = accuracy
w = np.array([0.5, 0.5])
stepratio = 0.1#0.02#(0.01/0.6)
direction_limit = 3
iteration_limit = 10
testvairable = 1
y = find_weights_joint(training, features, classes, labels, crossvalidationdiv, k_fold_iterations, i, w, stepratio, k, method1,
                       method2, direction_limit, iteration_limit, testvairable)
print(y)



