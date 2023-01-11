"""
Training classifiers
"""

from sklearn.decomposition import PCA, KernelPCA
import sklearn as sl
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import  StandardScaler

import pickle
import datetime
import pandas as pd
import numpy as np
import os
from itertools import product
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv("../data_processed/data.csv")
test_data = pd.read_csv("../data_processed/test.csv")

# Set labels to class, change string to 0 and 1
train_labels = train_data["class"]
train_labels.replace([" <=50K", " >50K"], [0, 1], inplace=True)
test_labels = test_data["class"]
test_labels.replace([" <=50K.", " >50K."], [0, 1], inplace=True)

full_data = False
features_to_drop = ["capital-gain", "marital-status"]

# OneHotEncoding for categorical input. Maybe not for "education", since we have "education-num"
# Combine train and test for OneHotEncoding, label them for splitting later
train_data["train"] = 1
test_data["train"] = 0
data = pd.concat([train_data, test_data], ignore_index=True)
data.drop("education", axis=1, inplace=True)
if not full_data:
    data.drop(features_to_drop, axis=1, inplace=True)
input_data = data.loc[:, data.columns != 'class']
categorical_columns = [input_data.columns[i] for i in range(len(input_data.columns)) if
                       input_data.dtypes[i] == "object"]
transformer = make_column_transformer((OneHotEncoder(sparse_output=False), categorical_columns),
                                      remainder="passthrough")
transformed = transformer.fit_transform(input_data)
input_data = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

# recreate train and test input sets
input_data.drop("remainder__fnlwgt", axis=1, inplace=True)
train_input = input_data[input_data["remainder__train"] == 1]
test_input = input_data[input_data["remainder__train"] == 0]
train_input = train_input.drop("remainder__train", axis=1)
test_input = test_input.drop("remainder__train", axis=1)

# classifier to train
classifier = "GradientBoost"
# Train and evaluate
# 1 Nearest Neighbour: 0.97 accuracy => Same input, different labels occur sometimes.
# clf_base = KNeighborsClassifier(n_neighbors=1)
# clf_base.fit(train_input,train_labels)
# print(clf_base.score(train_input,train_labels))

# RandomForest
# clf = RandomForestClassifier()
# params = {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 8, 10, 15], "n_estimators": [50, 100, 200, 500],
#          "class_weight": [dict(data["fnlwgt"]), None, "balanced"], "max_features": ["sqrt", "log2", 20],
#          "random_state": [0]}

# GradientBoost
clf = GradientBoostingClassifier()
params = {"loss": ["log_loss", "exponential"], "learning_rate": [0.01, 0.1, 1],
"n_estimators": [50, 100, 200, 500],
"criterion": ["friedman_mse", "squared_error"], "max_features": [None, "sqrt", "log2", 20],
"random_state": [0]}

if classifier == "SVM":
    # SVM
    #clf = SVC(kernel="linear")
    # best_params for "2023_01_09_15_00_37_LinearSVC"
    best_params = {'C': 0.1, 'dual': False, 'loss': 'squared_hinge', 'max_iter': 100, 'penalty': 'l2', 'random_state': 0, 'tol': 1e-05}
    clf = LinearSVC()
    params = {"C": [0.01, 0.1, 1, 10],
              "class_weight": [dict(data["fnlwgt"]),None,"balanced"],
              "penalty" : ["l2"],
              "loss" :["hinge","squared_hinge"],
              "random_state" : [0],
              "dual" : [True, False],
              "tol" : [1e-5, 1e-4, 1e-3],
              "max_iter" : [100,1000,10000],
              }
    sc = StandardScaler()
    train_input = sc.fit_transform(train_input)
    # Create the Nystroem transformer

    transformer = Nystroem(kernel='rbf', n_components=100, random_state=42, n_jobs=-1)
    # Transform the training data using the Nystroem transformer
    #train_input = transformer.fit_transform(train_input, train_labels)
    print(train_input.shape)
    #print(train_input.shape)
    # train_input = np.dot(transformed_input,transformed_input.T)
    pca = PCA(n_components=20)
    #train_input = pca.fit_transform(train_input)
    #kernel_pca = KernelPCA(n_components=10, kernel="rbf", gamma=10, fit_inverse_transform=False,alpha=0.1, n_jobs=-1)
    #train_input = kernel_pca.fit_transform(train_input)
    print(train_input.shape)

print("Start fitting")
grid_clf = GridSearchCV(clf, params, n_jobs=-1, cv=5, scoring="precision", verbose=4)  # default is 5-fold CV
grid_clf.fit(train_input, train_labels)
# display = LearningCurveDisplay.from_estimator(SVC(kernel="linear"), train_input, train_labels,
#                                              train_sizes=[50, 100, 200, 500, 1000],verbose=3, cv=5, n_jobs=-1, scoring="accuracy")
# display.plot()
# plt.show()
print("Fit done")
if classifier == "SVM":
    # Transform the test data using the Nystroem transformer
    # test_input = transformer.transform(test_input)
    # test_input = np.dot(test_input, test_input.T)
    # test_input = kernel_pca.transform(test_input)
    test_input = sc.transform(test_input)
    #test_input = pca.transform(test_input)
    pass

print(grid_clf.best_params_)
print(grid_clf.score(test_input, test_labels))

dir_name = "../models/" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + type(clf).__name__
os.mkdir(dir_name)
with open(dir_name + "/params.pickle", "wb") as file:
    pickle.dump(params, file)
with open(dir_name + "/best_params.pickle", "wb") as file:
    pickle.dump(grid_clf.best_params_, file)
with open(dir_name + "/input_data.pickle", "wb") as file:
    pickle.dump(input_data, file)
with open(dir_name + "/model", "wb") as file:
    pickle.dump(grid_clf, file)


