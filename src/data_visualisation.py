import seaborn as sns
import pandas as pd
import sklearn
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, CategoricalColorMapper
from sklearn.decomposition import TruncatedSVD, KernelPCA
from bokeh.models.tickers import FixedTicker
from math import pi
from bokeh.io import export_png
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, class_likelihood_ratios, f1_score, balanced_accuracy_score, recall_score
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
                                     np.arange(0, 100000, 10000)],
                    "marital-status": [np.arange(7), 40000,
                                       np.arange(7)]
                    }

    if feature == "race" or feature == "marital-status":
        sorting_index = data[feature].value_counts().index
        barlist = ax.bar(x=sorting_index, height=data[feature].value_counts().reindex(index=sorting_index),
                         label=feature_name,
                         align="center", width=0.8,
                         color="grey", alpha=0.9,
                         edgecolor="xkcd:almost black", linewidth=1)

    else:
        bin_entries, hist_bins, barlist = ax.hist(data[feature], bins=feature_dict[feature][0], label=feature_name,
                                                  align="mid", rwidth=0.8,
                                                  color="grey", alpha=0.9,
                                                  edgecolor="xkcd:almost black", linewidth=1)

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
    if feature == "marital-status":
        ax.set_xticklabels(data[feature].value_counts().index, rotation=60, ha="right", fontsize=7)

    plt.savefig(f"../graphics/{feature}_plot.pdf")

    # classes
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    if feature == "race" or feature == "marital-status":
        ax.bar(x=sorting_index,
               height=data[feature][all_labels == 1].value_counts().reindex(index=sorting_index),
               label="Income > 50k",
               align="center", width=0.8,
               color="xkcd:navy green", alpha=0.9,
               edgecolor="xkcd:almost black", linewidth=1)

    else:
        ax.hist(data[feature][all_labels == 1], bins=feature_dict[feature][0], label="Income > 50k", align="mid",
                rwidth=0.8, color="xkcd:navy green", alpha=0.9, edgecolor="xkcd:almost black", linewidth=1)

    for bar in barlist:
        bar.set_color("xkcd:light brown")
        bar.set_edgecolor("black")
        bar.set_linewidth(1)

    bar_legend = ax.legend(["$\leq$50k\$ p.a.", " $>$50k\$ p.a."], loc="upper left", title="Income class",
                           facecolor="whitesmoke",
                           edgecolor="xkcd:almost black", handlelength=3, handleheight=1.5, title_fontsize="small")

    plt.savefig(f"../graphics/{feature}_classes_plot.pdf")

    # probability
    ax2 = plt.twinx()
    ax2.set_ylabel('Probability')
    if feature == "education-num":
        ax2.set_yticks(np.linspace(0, 1, 5))
        # ax2.set_yticklabels(np.linspace(0, 1, 5))
    if feature == "marital-status":
        ax2.set_yticks(np.linspace(0, 1, 5))
        # ax2.set_yticklabels(np.linspace(0, 1, 5))
    ax2.set_ylim(0, 1)

    values = train_input[feature].value_counts()
    values_class_1 = train_input[feature][train_labels == 1].value_counts()

    if feature == "race" or feature == "marital-status":
        probabilities = values_class_1 / values
        prob_line = ax2.scatter(x=sorting_index, y=probabilities.reindex(index=sorting_index), c="firebrick", s=10,
                                label="Real probability", marker="8")
    else:
        values = values.sort_index(ascending=True)
        values_class_1 = values_class_1.sort_index(ascending=True)
        probabilities = values_class_1 / values
        probabilities = probabilities.interpolate(method="linear")
        prob_line, = ax2.plot(probabilities.index, probabilities.values, "firebrick",
                              label="Real probability")

    handles = [prob_line]
    ax2.legend(handles=handles, loc="upper right", facecolor="whitesmoke",
               edgecolor="xkcd:almost black", handlelength=2.6, handleheight=1.5)

    plt.savefig(f"../graphics/{feature}_classes_probability_plot.pdf")

    # feature dependence
    unique_values = list(train_input[feature].unique())
    unique_values.sort()

    clf.predict_proba = clf.predict

    models_new = []
    models_new.append("../models/" + "2023_01_23_06_19_00_LinearSVC")
    models_new.append("../models/" + "2023_01_23_02_48_17_LGBMClassifier")

    colors = ["darkcyan","rebeccapurple"]
    for k, model_path_new in enumerate(models_new):
        if "LGBM" in model_path_new or "Gradient" in model_path_new:
            model_name_new = "Gradient Boost"
        if "LinearSVC" in model_path_new:
            model_name_new = "Linear SVC"
        if "RandomForest" in model_path_new:
            model_name_new = "Random Forest"
        with open(model_path_new + "/model", "rb") as file:
            clf_new = pickle.load(file)
        clf_new.predict_proba = clf_new.predict
        results = partial_dependence(clf_new, train_input, [feature],
                                     grid_resolution=200,
                                     response_method="predict_proba")

        # feature dependence
        if feature == "race" or feature == "marital-status":
            pred_line = ax2.scatter(x=results["values"][0], y=results["average"][0], c=colors[k], marker="8", s=10,
                                    label=f"Learned dependence \nof {model_name_new}")
        else:
            pred_line, = ax2.plot(results["values"][0], results["average"][0], colors[k],
                                  label=f"Learned dependence \nof {model_name_new}")

        handles = handles + [pred_line]
        ax2.legend(handles=handles, loc="upper right", facecolor="whitesmoke",
                   edgecolor="xkcd:almost black", handlelength=2.6, handleheight=1.5)

        plt.savefig(f"../graphics/{feature}_{model_name_new}_feature_dependence.pdf")


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


def plot_feature_importance(models):
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")

    ax.set_xlabel("Features")
    ax.set_ylim(-0.003, 0.1)
    df = pd.DataFrame()
    for i, model_path in enumerate(models):
        if "LGBM" in model_path or "Gradient" in model_path:
            model_name = "Gradient Boost"
        if "LinearSVC" in model_path:
            model_name = "Linear SVC"
        if "RandomForest" in model_path:
            model_name = "Random Forest"
        with open(model_path + "/model", "rb") as file:
            clf = pickle.load(file)

        result = permutation_importance(clf, test_input, test_labels, n_repeats=1, random_state=0, n_jobs=-1,scoring="balanced_accuracy")
        importances = pd.Series(result.importances_mean, index=train_input.columns)
        df[model_name] = importances

    df.plot.bar(ax=ax, color=["darkcyan","rebeccapurple"], alpha=0.8, edgecolor="xkcd:almost black", linewidth=1)
    ax.grid(axis="y", linestyle="dashed", color="gray")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(len(train_input.columns)))
    ax.set_ylabel("Mean balanced accuracy decrease")
    tick_labels = [str(col).title() for col in train_input.columns]
    ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=7)
    plt.savefig(f"../graphics/feature_importances_combined.pdf")


def plot_race_education():
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.05, .78, 0.95])

    bins = [[0, 8], [9, 9], [10, 10], [11, 12], [13, 16]]
    final_percentages = np.zeros((5, 6))
    value_counts = data['education-num'].value_counts(normalize=True)
    for ind, it in value_counts.items():
        for i, bin in enumerate(bins):
            if ind >= bin[0] and ind <= bin[1]:
                final_percentages[i][0] += it

    for race_ind, race in enumerate(data["race"].unique()):
        value_counts = data['education-num'][data["race"] == race].value_counts(normalize=True)
        for ind, it in value_counts.items():
            for i, bin in enumerate(bins):
                if ind >= bin[0] and ind <= bin[1]:
                    final_percentages[i][race_ind + 1] += it

    cmap = mpl.colormaps["Set2"]
    handles = []
    bottom = np.zeros(6)
    educ_labels = ["Less than \nhigh school", "High school only", "College only", "Associates Degree \n only",
                   "Bachelor's or \nhigher degree"]
    for i in range(5):
        print(cmap(i))
        this_handle = ax.bar(x=range(6), height=final_percentages[i, :], bottom=bottom, width=0.8, align="center",
                             edgecolor="xkcd:almost black", linewidth=1, label=educ_labels[i], color=cmap(i))
        handles.append(this_handle)

        for j, rect in enumerate(this_handle):
            height = rect.get_height()
            ax.text(x=rect.get_x() + rect.get_width() / 2, y=-0.013 + bottom[j] + height / 2,
                    s=str(int(round(final_percentages[i, j] * 100, 0))), ha="center")

        bottom = final_percentages[i, :] + bottom

    handles.reverse()
    bar_legend = ax.legend(handles=handles, loc="upper left", title="Education/Degree",
                           facecolor="whitesmoke",
                           edgecolor="xkcd:almost black", handlelength=3, handleheight=1.5, title_fontsize="small",
                           bbox_to_anchor=(1, 1.015),
                           fontsize=8)

    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")
    ax.set_axisbelow(True)
    ax.set_ylabel("Percent")
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels([0,20,40,60,80,100])
    ax.set_xlabel("Race/Ethnicity")

    ax.set_xticks(range(6))
    ax.set_xticklabels(["All", "White", "Black", "Asian", "Native", "Other"])
    plt.savefig(f"../graphics/race_education.pdf")



def plot_classes():
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.85])
    labels = ["Yearly Income of $\leq$50k\$", "Yearly Income of $>$50k\$"]
    label_count = data["class"].value_counts()
    ax.pie(label_count.values, explode=(0, 0.3), labels=labels, colors=["xkcd:light brown", "xkcd:navy green"],
           autopct='%1.1f%%', wedgeprops={'linewidth': 2, "edgecolor": "xkcd:almost black", "alpha": 0.9},
           startangle=135,
           labeldistance=1.2,
           textprops=dict(va='center', ha='center', fontsize=12, style="oblique"))

    ax.set_aspect("equal")
    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")
    ax.set_axisbelow(True)
    plt.savefig(f"../graphics/classes.pdf")

def plot_metrics(models):
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.04, 0.95, 0.96])
    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")
    #ax.set_xlabel("Scoring metric")
    ax.set_ylim(0, 1)
    df = pd.DataFrame()
    for i, model_path in enumerate(models):
        if "LGBM" in model_path or "Gradient" in model_path:
            model_name = "Gradient Boost"
        if "LinearSVC" in model_path:
            model_name = "Linear SVC"
        if "RandomForest" in model_path:
            model_name = "Random Forest"
        with open(model_path + "/model", "rb") as file:
            clf = pickle.load(file)
        scores = []
        pred_labels = clf.best_estimator_.predict(test_input)
        scores.append(balanced_accuracy_score(test_labels, pred_labels))
        scores.append(accuracy_score(test_labels, pred_labels))
        scores.append(recall_score(test_labels, pred_labels))

        scores_ds = pd.Series(scores)
        df[model_name] = scores_ds
    print(df)
    df.plot.bar(ax=ax, color=["darkcyan","royalblue","rebeccapurple",], alpha=0.8, edgecolor="xkcd:almost black", linewidth=1)
    ax.grid(axis="y", linestyle="dashed", color="gray")
    ax.set_axisbelow(True)
    #ax.set_xticks(range(3))
    ax.set_ylabel("Score")
    tick_labels = ["Balanced accuracy","Accuracy","Recall"]
    ax.set_xticklabels(tick_labels, ha="center", fontsize=10,rotation = 0)
    plt.savefig(f"../graphics/metrics_combined.pdf")

if __name__ == "__main__":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()
    plot_race_education()
    """
    # plot_classes()
    features = [
        "race",
        "age", "education-num",
        "marital-status"
    ]
    models = []
    models.append("../models/" + "2023_01_23_06_19_00_LinearSVC")
    #models.append("../models/" + "2023_01_23_06_14_16_RandomForestClassifier")
    models.append("../models/" + "2023_01_23_02_48_17_LGBMClassifier")
    #plot_feature_importance(models)
    plot_metrics(models)
    for model_path in models:
        if "LGBM" in model_path or "Gradient" in model_path:
            model_name = "Gradient Boost"
        if "LinearSVC" in model_path:
            model_name = "Linear SVC"
        if "RandomForest" in model_path:
            model_name = "Random Forest"
        print(model_name)
        with open(model_path + "/model", "rb") as file:
            clf = pickle.load(file)

        for feature in features:
            print(feature)
            plot_feature(feature, model_name)
        # plot_PCA(train_input)

        # sns_plot = sns.pairplot(data, hue="class",diag_kind="hist")
        # sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
        # sns_plot.savefig("data_plot.png")"""