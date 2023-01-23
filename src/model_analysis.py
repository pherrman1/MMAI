from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.svm import LinearSVC
from sklearn.model_selection import validation_curve

import data_preprocessing as dp

model_path = "../models/" + "2022_12_28_13_32_58_RandomForestClassifier"
model_path = "../models/" + "2023_01_09_15_00_37_LinearSVC"
# model_path = "../models/" + "2023_01_09_17_07_49_LinearSVC"


models = []
# models.append("../models/" + "2023_01_23_06_14_16_RandomForestClassifier")
models.append("../models/" + "2023_01_23_02_48_17_LGBMClassifier")
#models.append("../models/" + "2023_01_23_06_19_00_LinearSVC")

for model_path in models:
    # load model and datasets
    with open(model_path + "/model", "rb") as file:
        clf = pickle.load(file)
    print(model_path)
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()
    param_range = np.arange(-1, 100, 1)
    train_scores, test_scores = validation_curve(clf.best_estimator_, train_input, train_labels, param_name="LGBM__max_depth",
                                                 param_range=param_range, verbose=5, n_jobs=-1,
                                                 scoring="balanced_accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVC")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()
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
