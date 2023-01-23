import seaborn as sns
import pandas as pd
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, CategoricalColorMapper
from sklearn.decomposition import TruncatedSVD, KernelPCA
from bokeh.models.tickers import FixedTicker
from math import pi
from bokeh.io import export_png
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.neighbors import KernelDensity
import data_preprocessing as dp
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import permutation_importance, partial_dependence
import numpy as np
import pandas as pd
from sklearn.utils import validation
import matplotlib as mpl

from matplotlib import rcParams
from sklearn.decomposition import PCA

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

mpl.rcParams.update(nice_fonts)
mpl.rcParams["figure.dpi"] = 200


def plot_corr_matr(data):
    data['sex'].replace([' Male', ' Female'], [0, 1], inplace=True)

    # compute Pearson correlation
    corr = data.corr()
    variables = list(corr)
    print(variables)
    # linearize the correlation matrix
    lin_corr = pd.melt(corr.assign(index=corr.index), id_vars=['index'])
    lin_corr['size'] = [abs(val) * 50 for val in lin_corr.value]

    p = figure(x_range=variables, y_range=variables, width=800, height=700,
               title="Correlation matrix")

    p.square(source=lin_corr, x='index', y='variable', size='size', color=linear_cmap('value', 'RdYlBu7', -1, 1))

    color_bar = ColorBar(color_mapper=LinearColorMapper('RdYlBu5', low=-1, high=1),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    p.xgrid.ticker = FixedTicker(ticks=list(range(1, len(corr))))
    p.ygrid.ticker = FixedTicker(ticks=list(range(1, len(corr))))
    p.xaxis.major_label_orientation = -pi / 4
    show(p)
    export_png(p, filename="correlation_numeric_features.png")


def plot_feature(feature, model_name):
    feature_name = feature.split("_")[-1].title()

    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.15, .84, 0.95])

    # plot feature histogram
    feature_dict = {"age": [np.arange(15, 91, 5), 10000, np.arange(15, 91, 5)],
                    "education-num": [np.arange(0.5, 17.5, 1), 20000, np.arange(1, 17, 1)],
                    "occupation": [None, 10000, np.arange(len(data[feature].unique()))],
                    "hours-per-week": [np.linspace(0, 100, 20), 25000, np.linspace(0, 100, 11)],
                    "race": [None, 50000, np.arange(5)],
                    "capital-gain": [np.arange(0, 100000, 10000), 50000,
                                     np.arange(0, 100000, 10000)]
                    }

    if feature == "race" or feature == "occupation":
        sorting_index = data[feature].value_counts().index
        bin_entries = ax.bar(x=sorting_index, height=data[feature].value_counts().reindex(index=sorting_index),
                             label=feature_name,
                             align="center", width=0.8,
                             color="xkcd:light brown", alpha=0.9,
                             edgecolor="dimgrey", linewidth=1.5)

    else:
        bin_entries, hist_bins, patches = ax.hist(data[feature], bins=feature_dict[feature][0], label=feature_name,
                                                  align="mid", rwidth=0.8,
                                                  color="xkcd:light brown", alpha=0.9,
                                                  edgecolor="dimgrey", linewidth=1.5)

    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")
    ax.grid(axis="y", linestyle="dashed", color="gray")
    ax.set_axisbelow(True)
    ax.set_ylabel("People count")
    ax.set_xlabel(feature_name)
    ax.set_ylim(0, feature_dict[feature][1])
    ax.set_xticks(feature_dict[feature][2])

    if feature == "education-num":
        train_data = pd.read_csv("../data_processed/data.csv")
        index_educ = np.argsort(train_data["education-num"].unique())
        x_tick_educ = np.array(train_data["education"].unique())[index_educ]
        ax.set_xticklabels(x_tick_educ, rotation=60, ha="right")
    if feature == "race":
        ax.set_xticklabels(["White", "Black", "Asian", "Native", "Other"])
    if feature == "occupation":
        ax.set_xticklabels(data[feature].value_counts().index, rotation=60, ha="right", fontsize=7)

    plt.savefig(f"../graphics/{feature}_plot.pdf")

    # classes
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    if feature == "race" or feature == "occupation":
        ax.bar(x=sorting_index,
               height=data[feature][all_labels == 1].value_counts().reindex(index=sorting_index),
               label="Income > 50k",
               align="center", width=0.8,
               color="xkcd:navy green", alpha=0.9,
               edgecolor="dimgrey", linewidth=1.5)
    else:
        ax.hist(data[feature][all_labels == 1], bins=feature_dict[feature][0], label="Income > 50k", align="mid",
                rwidth=0.8, color="xkcd:navy green", alpha=0.7, edgecolor="dimgrey", linewidth=1.5)

    bar_legend = ax.legend(["$\leq$50k\$ p.a.", " $>$50k\$ p.a."], loc="upper left", title="Income class",
                           facecolor="whitesmoke",
                           edgecolor="black", handlelength=3, handleheight=1.5, title_fontsize="small")
    plt.savefig(f"../graphics/{feature}_classes_plot.pdf")

    # probability
    ax2 = plt.twinx()
    ax2.set_ylabel('Probability')
    if feature == "education-num":
        ax2.set_yticks(np.linspace(0, 1, 9))
    ax2.set_ylim(0, 1)

    values = train_input[feature].value_counts()
    values_class_1 = train_input[feature][train_labels == 1].value_counts()

    if feature == "race" or feature == "occupation":
        probabilities = values_class_1 / values
        prob_line = ax2.scatter(x=sorting_index, y=probabilities.reindex(index=sorting_index), c="royalblue", s=10,
                                label="Real probability", marker="8")
    else:
        values = values.sort_index(ascending=True)
        values_class_1 = values_class_1.sort_index(ascending=True)
        probabilities = values_class_1 / values
        probabilities = probabilities.interpolate(method="linear")
        prob_line, = ax2.plot(probabilities.index, probabilities.values, "royalblue",
                              label="Real probability")

    handles = [prob_line]
    ax2.legend(handles=handles, loc="upper right", facecolor="whitesmoke",
               edgecolor="black", handlelength=2.6, handleheight=1.5)

    plt.savefig(f"../graphics/{feature}_classes_probability_plot.pdf")

    # feature dependence
    unique_values = list(train_input[feature].unique())
    unique_values.sort()

    clf.predict_proba = clf.predict
    print("calc partial_dependence")
    results = partial_dependence(clf, train_input, [feature],
                                 # categorical_features=["workclass", "marital-status", "occupation", "relationship",
                                 #                      "race", "sex", "native-country"],
                                 grid_resolution=200,
                                 response_method="predict_proba")

    # feature dependence
    if feature == "race" or feature == "occupation":
        pred_line = ax2.scatter(x=results["values"][0], y=results["average"][0], c="firebrick", marker="8", s=10,
                                label=f"Learned dependence \nof {model_name}")
    else:
        pred_line, = ax2.plot(results["values"][0], results["average"][0], "firebrick",
                              label=f"Learned dependence \nof {model_name}")
    handles = [prob_line, pred_line]
    ax2.legend(handles=handles, loc="upper right", facecolor="whitesmoke",
               edgecolor="black", handlelength=2.6, handleheight=1.5)

    plt.savefig(f"../graphics/{feature}_{model_name}_feature_dependence.pdf")


def plot_PCA(train_input):
    # feature types
    numeric_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
                            "native-country"]
    # Preprocessor transformer: Numeric features get standardized, categorical get onehotencoded
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], sparse_threshold=0,
    )

    X = preprocessor.fit_transform(train_input)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())

    gammas = [0.001, 0.002]

    """fig, ax = plt.subplots(len(gammas), figsize=(20, 15))
    for i, gamma in enumerate(gammas):
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        X_pos = X[train_labels == 1]
        X_neg = X[train_labels == 0]"""
    fig, ax = plt.subplots(1)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    X_pos = X[train_labels == 1]
    X_neg = X[train_labels == 0]
    for X_class, color in [(X_pos, "royalblue"), (X_neg, "firebrick")]:
        ax.scatter(X_class[:, 0], X_class[:, 1], color=color, s=0.1)

    for i, (x, y) in enumerate(zip(*pca.components_)):
        if i < 2 and i >= 0:
            [[x, y]] = normalize(np.array([x, y]).reshape(1, -1))
            x, y = x * 5, y * 5
            ax.plot([0, x], [0, y])
            ax.text(x, y, s=pca.feature_names_in_[i])

    plt.savefig(f"../graphics/pca.png")


def plot_feature_importance(model_name):
    result = permutation_importance(clf, test_input, test_labels, n_repeats=1, random_state=0, n_jobs=-1)

    forest_importances = pd.Series(result.importances_mean, index=train_input.columns)

    # use all onehotencoder features
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_ylabel("Mean accuracy decrease")
    plt.savefig(f"../graphics/{model_name}_feature_importance.png")

def plot_race_education():
    pass

if __name__ == "__main__":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()
    features = ["occupation","age","education-num","race"]
    models = []
    models.append("../models/" + "2023_01_23_01_56_14_LinearSVC")
    models.append("../models/" + "2023_01_23_00_27_31_LGBMClassifier")
    models.append("../models/" + "2023_01_22_23_47_08_RandomForestClassifier")
    for model_path in models:
        print(model_name)
        if "LGBM" in model_path or "Gradient" in model_path:
            model_name = "Gradient Boost"
        if "LinearSVC" in model_path:
            model_name = "Linear SVC"
        if "RandomForest" in model_path:
            model_name = "Random Forest"
        print(model_name)
        with open(model_path + "/model", "rb") as file:
            clf = pickle.load(file)

        plot_feature_importance(model_name)

        for feature in features:
            print(feature)
            plot_feature(feature, model_name)
        # plot_PCA(train_input)

        # sns_plot = sns.pairplot(data, hue="class",diag_kind="hist")
        # sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
        # sns_plot.savefig("data_plot.png")
