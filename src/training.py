"""
Training classifiers
"""

import sklearn as sl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

import pandas as pd
import numpy as np
import os
from itertools import product

# Load data
train_data = pd.read_csv("../data_processed/data.csv")
test_data = pd.read_csv("../data_processed/test.csv")

# Set labels to class, change string to 0 and 1
train_labels = train_data["class"]
train_labels.replace([" <=50K", " >50K"], [0, 1], inplace=True)
test_labels = test_data["class"]
test_labels.replace([" <=50K.", " >50K."], [0, 1], inplace=True)

# OneHotEncoding for categorical input. Maybe not for "education", since we have "education-num"
# Combine train and test for OneHotEncoding, label them for splitting later
train_data["train"] = 1
test_data["train"] = 0
data = pd.concat([train_data, test_data], ignore_index=True)
data.drop("education",axis=1,inplace=True)
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

# Train and evaluate
rf = RandomForestClassifier()
params = {"criterion": ["gini", "entropy"], "max_depth": [None, 3, 5, 8], "n_estimators": [10, 50, 100, 200, 500],
          "class_weight": [dict(data["fnlwgt"]), None, "balanced"], "random_state": [0]}

clf = GridSearchCV(rf, params, n_jobs=-1, cv=5, scoring="accuracy")  # default is 5-fold CV
clf.fit(train_input, train_labels)
# print(clf.cv_results_)
print(clf.score(test_input, test_labels))
print(clf.best_params_)