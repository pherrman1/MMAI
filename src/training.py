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

import pandas as pd
import numpy as np
import os
from itertools import product

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

train_data = pd.read_csv("../data_processed/data.csv")
train_labels = train_data["class"]
train_data = train_data.iloc[: , 1:-1]

train_data["workclass"].replace([" Private", " Self-emp-not-inc", " Self-emp-inc", " Federal-gov", " Local-gov", " State-gov", " Without-pay", " Never-worked"], [0,1,2,3,4,5,6,7], inplace = True)


rf = RandomForestClassifier()
params = {"criterion": ["gini", "entropy"], "max_depth": [None, 3, 5, 8], "n_estimators": [10, 50, 100, 200], "class_weight": [None, "balanced"], "random_state": [0]}


clf = GridSearchCV(rf, params) #default is 5-fold CV
clf.fit(train_data, train_labels)


