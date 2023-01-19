
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import data_preprocessing as dp

model_path = "../models/" + "2022_12_28_13_32_58_RandomForestClassifier"
model_path = "../models/" + "2023_01_09_15_00_37_LinearSVC"
#model_path = "../models/" + "2023_01_09_17_07_49_LinearSVC"

model_path =  "../models/" + "2023_01_03_15_10_11_GradientBoostingClassifier"

# load model and datasets
with open(model_path + "/input_data.pickle", "rb") as file:
    input_data = pickle.load(file)
with open(model_path + "/model", "rb") as file:
    clf = pickle.load(file)

print(input_data)

train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()



pred_labels = clf.best_estimator_.predict(test_input)
print(accuracy_score(test_labels,pred_labels))
print(clf.best_params_)

"""
# print(len(test_labels[test_labels == 1]) / len(test_labels))

# continuous features
cont_features = [i for i in train_input.columns if i.__contains__("remainder")]
#cont_features.remove('remainder__capital-gain') #this is somehow not working



#plot_feature_dependence_continuous("remainder__capital-gain")

#def plot_validation_curve(feature, param_range):

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
ax.set_title("Feature importances using permutation on full model " +  model_path.split("_")[-1])
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
"""