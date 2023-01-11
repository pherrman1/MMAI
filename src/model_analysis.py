from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

model_path = "../models/" + "2022_12_28_13_32_58_RandomForestClassifier"
model_path = "../models/" + "2023_01_09_15_00_37_LinearSVC"
# model_path =  "../models/" + "2023_01_03_15_10_11_GradientBoostingClassifier"

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

pred_labels = clf.predict(test_input)
print(precision_score(test_labels,pred_labels))
print(clf.best_params_)

# print(len(test_labels[test_labels == 1]) / len(test_labels))

# continuous features
cont_features = [i for i in train_input.columns if i.__contains__("remainder")]
#cont_features.remove('remainder__capital-gain') #this is somehow not working


def plot_feature_dependence_continuous(feature):
    unique_values = list(train_input[feature].unique())
    unique_values.sort()
    value_occurences = [len(train_input[(train_input[feature] == value)])
                        for value in unique_values]
    print(value_occurences)
    print(sum(value_occurences))
    feature_dependence = [len(train_input[(train_input[feature] == value) & (train_labels == 1)]) /
                          len(train_input[(train_input[feature] == value)])
                          for value in unique_values]

    fig, ax = plt.subplots()
    results = partial_dependence(clf, train_input, [feature], grid_resolution=len(unique_values))
    ax.plot(unique_values, results["average"][0], "k")
    m = ax.scatter(unique_values, feature_dependence, c=value_occurences, cmap="cool")
    plt.colorbar(mappable=m, ax=ax, location = "right", label = "frequency of values")
    ax.set_title(feature.split("_")[-1] + " " + model_path.split("_")[-1])
    plt.show()
    plt.savefig(f"{features}_plot.png")

#plot_feature_dependence_continuous("remainder__capital-gain")

#def plot_validation_curve(feature, param_range):

# Feature importance
result = permutation_importance(clf, test_input, test_labels, n_repeats=1, random_state=0, n_jobs=-1)

forest_importances = pd.Series(result.importances_mean, index=train_input.columns)
"""
# use all onehotencoder features
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
plt.show()
"""
# back to original features by summing their means
features = ["age", "workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

for feature in features:
    col_list = [i for i in train_input.columns if i.__contains__("_" + feature)]
    forest_importances[feature] = forest_importances[col_list].abs().sum(axis=0)
    forest_importances = forest_importances.drop(col_list)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model " +  model_path.split("_")[-1])
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()