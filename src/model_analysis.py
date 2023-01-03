from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

model_path = "../models/" + "2022_12_28_13_32_58_RandomForestClassifier"

# load model and datasets
with open(model_path + "/input_data.pickle", "rb") as file:
    input_data = pickle.load(file)
with open(model_path + "/model", "rb") as file:
    clf = pickle.load(file)

train_input = input_data[input_data["remainder__train"] == 1]
test_input = input_data[input_data["remainder__train"] == 0]
train_input = train_input.drop("remainder__train", axis=1)
test_input = test_input.drop("remainder__train", axis=1)

# Load data
train_data = pd.read_csv("../data_processed/data.csv")
test_data = pd.read_csv("../data_processed/test.csv")

# Set labels to class, change string to 0 and 1
train_labels = train_data["class"]
train_labels.replace([" <=50K", " >50K"], [0, 1], inplace=True)
test_labels = test_data["class"]
test_labels.replace([" <=50K.", " >50K."], [0, 1], inplace=True)

print(len(test_labels[test_labels == 1]) / len(test_labels))

unique_ages = list(train_input["remainder__age"].unique())
unique_ages.sort()
ages = [len(train_input[(train_input["remainder__age"] == age)])
        for age in unique_ages]
print(ages)
print(unique_ages)
age_dependence = [len(train_input[(train_input["remainder__age"] == age) & (train_labels == 1)]) / len(
    train_input[(train_input["remainder__age"] == age)])
                  for age in unique_ages]

unique_ages = [age for age in unique_ages]
fig, ax = plt.subplots()
results = partial_dependence(clf, train_input, ["remainder__age"])
ax.plot(unique_ages, results["average"][0], "k")
ax.scatter( unique_ages, age_dependence, c = ages, cmap="cool")
plt.show()

"""
# Feature importance
result = permutation_importance(clf, test_input, test_labels, n_repeats=1, random_state=0, n_jobs=-1)

forest_importances = pd.Series(result.importances_mean, index=train_input.columns)

# use all onehotencoder features
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
plt.show()

# back to original features by summing their means
features = ["age", "workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

for feature in features:
    col_list = [i for i in train_input.columns if i.__contains__("_" + feature)]
    forest_importances[feature] = forest_importances[col_list].abs().sum(axis=0)
    forest_importances = forest_importances.drop(col_list)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
"""