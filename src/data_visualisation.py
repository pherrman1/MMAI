import seaborn as sns
import pandas as pd
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, CategoricalColorMapper
from bokeh.models.tickers import FixedTicker
from math import pi
from bokeh.io import export_png
import data_preprocessing as dp
import matplotlib.pyplot as plt
import pickle
from sklearn.inspection import permutation_importance, partial_dependence
import numpy as np
import pandas as pd
from sklearn.utils import validation

from matplotlib import rcParams


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


def plot_feature(feature):
    feature_name = feature.split("_")[-1].title()
    model_name = model_path.split("_")[-1].title()
    fig, ax = plt.subplots()
    fig.tight_layout(rect=[0.05, 0.05, .85, 0.95])
    # plot feature histogram
    bins = np.arange(15, 91, 5)
    bin_entries, hist_bins, patches = ax.hist(data[feature], bins=bins, label=feature_name, align="mid", rwidth=0.8,
                                              color="xkcd:light brown", alpha=0.9,
                                              edgecolor="grey", linewidth=1.5)
    fig.patch.set_facecolor("whitesmoke")
    ax.set_facecolor("snow")
    ax.grid(axis="y", linestyle="dashed", color="gray")
    ax.set_axisbelow(True)
    ax.set_ylabel("People Count")
    ax.set_xlabel(feature_name)
    ax.set_xticks(np.arange(15, 91, 5))
    ax.set_ylim(0, 10000)

    plt.savefig(f"../graphics/{feature}_{model_name}_plot.png", dpi=200)

    # classes
    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)
    ax.hist(data[feature][all_labels == 1], bins=bins, label="Income > 50k", align="mid",
            rwidth=0.8, color="xkcd:navy green", alpha=0.7, edgecolor="grey", linewidth=1.5)
    bar_legend = ax.legend(["$\leq$50k\$ p.a.", " >50k$ p.a."], loc="upper left", title="Income Class",
                           facecolor="whitesmoke",
                           edgecolor="black", handlelength=3, handleheight=1.5, title_fontsize="medium")
    plt.savefig(f"../graphics/{feature}_{model_name}_classes_plot.png", dpi=200)

    # feature dependence and probability
    feature_one_hot_name = "remainder__" + feature
    unique_values = list(train_input[feature_one_hot_name].unique())
    unique_values.sort()
    print("calc partial_dependence")
    results = partial_dependence(clf, train_input, [feature_one_hot_name], grid_resolution=100)
    print(results)
    ax2 = plt.twinx()

    values = train_input[feature_one_hot_name].value_counts().sort_index(ascending=True)
    values_class_1 = train_input[feature_one_hot_name][all_labels == 1].value_counts().sort_index(ascending=True)

    probabilities = values_class_1 / values
    probabilities = probabilities.interpolate(method="linear")

    prob_line, = ax2.plot(probabilities.index, probabilities.values, "royalblue",
                          label="Probability of original Dataset")

    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)

    #feature dependence and probability
    pred_line, = ax2.plot(results["values"][0], results["average"][0], "firebrick",
                          label=f"{feature_name} Dependence \nof {model_name}")
    handles = [pred_line, prob_line]
    ax2.legend(handles=handles, loc="upper right", facecolor="whitesmoke",
               edgecolor="black", handlelength=2.6, handleheight=1.5, scatteryoffsets=[0.5])

    plt.savefig(f"../graphics/{feature}_{model_name}_feature_dependence.png", dpi=200)


if __name__ == "__main__":
    train_input, train_labels, test_input, test_labels, fnlwgt, data = dp.get_data()
    feature = "age"
    model_path = "../models/" + "2023_01_09_15_00_37_LinearSVC"
    model_path = "../models/" + "2023_01_03_15_10_11_GradientBoostingClassifier"
    with open(model_path + "/model", "rb") as file:
        clf = pickle.load(file)

    plot_feature(feature)

    # sns_plot = sns.pairplot(data, hue="class",diag_kind="hist")
    # sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
    # sns_plot.savefig("data_plot.png")
