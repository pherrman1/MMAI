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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier
import data_preprocessing as dp
import pickle
import datetime
import pandas as pd
import numpy as np
import os
from itertools import product
import matplotlib.pyplot as plt


def transform(train_input, test_input, transformations):
    if transformations["standard_transform"]:
        # Standardize each feature
        sc = StandardScaler()
        train_input = sc.fit_transform(train_input)
        test_input = sc.transform(test_input)

    if transformations["nystroem_transformation"]:
        # Create the Nystroem transformer
        nystr = Nystroem(kernel='rbf', n_components=1000, random_state=42, n_jobs=-1)
        # Transform the training data using the Nystroem transformer
        train_input = nystr.fit_transform(train_input, train_labels)
        test_input = nystr.transform(test_input)
        print(train_input.shape)

    if transformations["pca_transformation"]:
        pca = PCA(n_components=20)
        train_input = pca.fit_transform(train_input)
        test_input = pca.transform(test_input)
        print(train_input.shape)

    if transformations["kernel_pca_transformation"]:
        kernel_pca = KernelPCA(n_components=10, kernel="rbf", gamma=10, fit_inverse_transform=False, alpha=0.1,
                               n_jobs=-1)
        train_input = kernel_pca.fit_transform(train_input)
        test_input = kernel_pca.transform(test_input)
        print(train_input.shape)

    return train_input, test_input


def grid_fit(clf, params, train_input, train_labels):
    print("Start GridSearch")
    grid_clf = GridSearchCV(clf, params, n_jobs=-1, cv=5, scoring="accuracy", verbose=4)  # default is 5-fold CV
    print("Start fitting")
    grid_clf.fit(train_input, train_labels)
    print("Fit done")
    print(grid_clf.best_params_)
    print(grid_clf.score(test_input, test_labels))
    return grid_clf


def save(clf, params, grid_clf, transformations):
    dir_name = "../models/" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + type(clf).__name__
    os.mkdir(dir_name)
    with open(dir_name + "/params.pickle", "wb") as file:
        pickle.dump(params, file)
    with open(dir_name + "/transformations.txt", "w") as file:
        for transform_str, transform_bool in transformations.items():
            file.write(transform_str + ": " + str(transform_bool))
            file.write('\n')
    with open(dir_name + "/model", "wb") as file:
        pickle.dump(grid_clf, file)


# display = LearningCurveDisplay.from_estimator(SVC(kernel="linear"), train_input, train_labels,
#                                              train_sizes=[50, 100, 200, 500, 1000],verbose=3, cv=5, n_jobs=-1, scoring="accuracy")
# display.plot()
# plt.show()


# classifier to train
classifier = "LGBM"

standard_transform = False
nystroem_transformation = False
pca_transformation = False
kernel_pca_transformation = False
transformations = {
    "standard_transform": standard_transform,
    "nystroem_transformation": nystroem_transformation,
    "pca_transformation": pca_transformation,
    "kernel_pca_transformation": kernel_pca_transformation}
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
if classifier == "GradientBoost":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data(oneHotEncoding=True)
    clf = GradientBoostingClassifier()
    best_params = {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_features': None,
                   'n_estimators': 500, 'random_state': 0}
    best_params = {v: [k] for v, k in best_params.items()}

    params = {"loss": ["log_loss", "exponential"], "learning_rate": [0.01, 0.1, 1],
              "n_estimators": [50, 100, 200, 500],
              "criterion": ["friedman_mse", "squared_error"], "max_features": [None, "sqrt", "log2", 20],
              "random_state": [0]}

    train_input, test_input = transform(train_input, test_input, transformations)
    grid_clf = grid_fit(clf, best_params, train_input, train_labels)
    save(clf, params, grid_clf, transformations)

if classifier == "LGBM":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data(oneHotEncoding=False)
    for feature in ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]:
        train_input[feature] = pd.Series(data[feature], dtype="category")
        test_input[feature] = pd.Series(data[feature], dtype="category")

    params = {
              "n_estimators": [50, 100, 200, 500],
              "max_depth": [-1, 5, 10],
              "learning_rate": [0.01, 0.1, 0.2],
              "random_state": [0],
              "num_leaves":[50,100,200],
              "n_jobs":[-1],
              "is_unbalance":[True,False]}

    clf = LGBMClassifier()
    grid_clf = grid_fit(clf, params, train_input, train_labels)
    best_params = {'categorical_feature': 'auto'}
    best_params = {v: [k] for v, k in best_params.items()}
    #clf.fit(train_input, train_labels)
    #print(clf.score(test_input, test_labels))


transformations = {}
if classifier == "SVM":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data(oneHotEncoding=True)
    # best_params for "2023_01_09_15_00_37_LinearSVC"
    best_params = {'C': [0.1], 'dual': [False], "class_weight": [fnlwgt, None, "balanced"], 'loss': ['squared_hinge'],
                   'max_iter': [100], 'penalty': ['l2'],
                   'random_state': [0], 'tol': [1e-05]}

    clf = LinearSVC()
    params = {"C": [0.01, 0.1, 1, 10],
              "class_weight": [fnlwgt, None, "balanced"],
              "penalty": ["l2"],
              "loss": ["hinge", "squared_hinge"],
              "random_state": [0],
              "dual": [True, False],
              "tol": [1e-5, 1e-4, 1e-3],
              "max_iter": [100, 1000, 10000],
              }
    train_input, test_input = transform(train_input, test_input, transformations)
    grid_clf = grid_fit(clf, best_params, train_input, train_labels)
    save(clf, params, grid_clf, transformations)
