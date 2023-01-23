"""
Training classifiers
"""
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sklearn as sl
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import learning_curve, HalvingGridSearchCV
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

# classifier to train
for classifier in ["LGBM",
                   "RandomForest","SVM"
                   ]:

    # transformations
    transformations = [Nystroem, PCA, KernelPCA]
    transformations_names = ["Nystroem", "PCA", "KernelPCA"]

    steps_transform = [(t_name, t) for t_name, t in zip(transformations_names, transformations)]

    clf_dict = {"LGBM": LGBMClassifier(),
                "RandomForest": RandomForestClassifier(),
                "SVM": LinearSVC()
                }
    clf = clf_dict[classifier]

    # get data
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()

    # feature types
    numeric_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Train and evaluate
    # 1 Nearest Neighbour: 0.97 accuracy => Same input, different labels occur sometimes.
    # clf_base = KNeighborsClassifier(n_neighbors=1)
    # clf_base.fit(train_input,train_labels)
    # print(clf_base.score(train_input,train_labels))

    # RandomForest

    if classifier == "RandomForest":
        # Final search: Fitting 5 folds for each of 960 candidates, totalling 4800 fits

        # best_params = {v: [k] for v, k in best_params.items()}
        # Maybe change class weight to none
        clf_params = {"criterion": ["log_loss", "gini"], "max_depth": [None, 5, 8, 10, 15],
                      "n_estimators": [50, 100, 200, 500],
                      "class_weight": [None, "balanced"], "max_features": ["sqrt", "log2", 10,20],
                      "random_state": [0]}

        scoring = "balanced_accuracy"
        transform_params = [{"Nystroem": ["passthrough"],
                             "PCA": ["passthrough", PCA(3), PCA(6), PCA(9)],
                             "KernelPCA": ["passthrough"],
                             }]

    # GradientBoost
    if classifier == "LGBM":
        best_params = {
                       'learning_rate': 0.1,
                       'max_depth': 8, 'metric': 'binary_logloss', 'n_estimators': 100,
                       'n_jobs': -1,
                       'num_leaves': 25, 'random_state': 0, 'scale_pos_weight': 2,
                       }

        best_params = {v: [k] for v, k in best_params.items()}

        clf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [-1, 5, 8],
            "learning_rate": [ 0.05, 0.1, 0.2],
            "random_state": [0],
            "num_leaves": [10, 25, 50],
            "n_jobs": [-1],
            "metric": ["binary_logloss", "binary_error", "average_precision"],
            "scale_pos_weight": [0.5, 1, 2, 4],
        }
        scoring = "balanced_accuracy"
        transform_params = [
            {"Nystroem": ["passthrough"],
             "PCA": ["passthrough", PCA(3), PCA(6), PCA(9)],
             "KernelPCA": ["passthrough"],
             }]

    if classifier == "SVM":
        # best_params for "2023_01_09_15_00_37_LinearSVC"
        best_params = {'C': [0.1], 'dual': [False], "class_weight": [None, "balanced"], 'loss': ['squared_hinge'],
                       'max_iter': [100], 'penalty': ['l2'],
                       'random_state': [0], 'tol': [1e-05]}

        clf_params = {"C": [0.01, 0.1, 1, 10],
                      "class_weight": [None, "balanced"],
                      "penalty": ["l2", "l1"],
                      "loss": ["hinge", "squared_hinge"],
                      "random_state": [0],
                      "dual": [False],
                      "tol": [1e-4, 1e-3, 1e-2],
                      "max_iter": [100, 1000, 10000],
                      }
        scoring = "balanced_accuracy"

        transform_params = [{"Nystroem": ["passthrough"],
                             "PCA": ["passthrough", PCA(3), PCA(6), PCA(9)],
                             "KernelPCA": ["passthrough"],
                             }, ]

    clf_params = {f"{classifier}__{k}": v for k, v in clf_params.items()}
    param_grid = [dict(trans_param, **clf_params) for trans_param in transform_params]

    # Preprocessor transformer: Numeric features get standardized, categorical get onehotencoded
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], remainder="passthrough", sparse_threshold=0
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor)] + steps_transform + [(classifier, clf)])

    print("Start GridSearch")
    # grid_search = HalvingGridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=1,
    #                           scoring=scoring, min_resources=180)
    grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=0, scoring=scoring)
    print("Start fitting")
    grid_search.fit(train_input, train_labels)
    print(pipe["preprocessor"].transformers)
    print("Fit done")
    print(grid_search.best_params_)
    final_score = grid_search.score(test_input, test_labels)
    print(final_score)

    dir_name = "../models/" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + type(clf).__name__
    os.mkdir(dir_name)
    with open(dir_name + "/params.pickle", "wb") as file:
        pickle.dump(clf_params, file)
    with open(dir_name + "/scoring.txt", "w") as file:
        file.write(f"{scoring}: {final_score}")
    with open(dir_name + "/model", "wb") as file:
        pickle.dump(grid_search, file)
    with open(dir_name + "/pipeline", "wb") as file:
        pickle.dump(pipe, file)
