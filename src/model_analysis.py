from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import data_preprocessing as dp

model_path = "../models/" + "2022_12_28_13_32_58_RandomForestClassifier"
model_path = "../models/" + "2023_01_09_15_00_37_LinearSVC"
# model_path = "../models/" + "2023_01_09_17_07_49_LinearSVC"

#model_path = "../models/" + "2023_01_20_16_53_31_GradientBoostingClassifier"
model_path = "../models/" + "2023_01_20_16_43_21_LGBMClassifier"
#model_path = "../models/" + "2023_01_20_18_54_24_LinearSVC"
models = []
#models.append("../models/" + "2023_01_23_01_56_14_LinearSVC")
models.append("../models/" + "2023_01_23_00_27_31_LGBMClassifier")
#models.append("../models/" + "2023_01_22_23_47_08_RandomForestClassifier")
for model_path in models:
    # load model and datasets
    with open(model_path + "/model", "rb") as file:
        clf = pickle.load(file)
    print(model_path)
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()

    print(clf.best_params_)
    pred_labels = clf.best_estimator_.predict(test_input)
    print(accuracy_score(test_labels,pred_labels))
    print(precision_score(test_labels, pred_labels))
    print(f1_score(test_labels, pred_labels))

"""
print(pred_labels)
print(confusion_matrix(test_labels, pred_labels, normalize="all"))
print(clf.best_params_)
final_score = clf.score(test_input, test_labels)
print(final_score)


week = (data[data["hours-per-week"]>80])
print(week.to_string())
print(sum(week)/len(week))

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
