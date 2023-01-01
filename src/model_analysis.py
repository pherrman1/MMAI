from sklearn.inspection import permutation_importance

model_path = "../model/"+ "2022_12_28_13_32_58_RandomForestClassifier"

# Feature importance
result = permutation_importance(clf, test_input, test_labels, n_repeats=10, random_state=0, n_jobs=1)
forest_importances = pd.Series(result.importances_mean, index=train_input.columns)

# use all onehotencoder features
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

# back to original features by summing their means
features = ["age", "workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

for feature in features:
    col_list = [i for i in train_input.columns if i.__contains__("_" + feature)]
    forest_importances[feature] = forest_importances[col_list].sum(axis=0)
    forest_importances = forest_importances.drop(col_list)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
