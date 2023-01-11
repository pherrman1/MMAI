import seaborn as sns
import pandas as pd
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, CategoricalColorMapper
from bokeh.models.tickers import FixedTicker
from math import pi


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


if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv("../data_processed/data.csv")
    test_data = pd.read_csv("../data_processed/test.csv")
    data = pd.concat([train_data, test_data], ignore_index=True)
    data["class"].replace([" <=50K", " >50K"], [0, 1], inplace=True)
    data["class"].replace([" <=50K.", " >50K."], [0, 1], inplace=True)
    data.drop("fnlwgt", axis=1, inplace=True)

    plot_corr_matr(data)

    # sns_plot = sns.pairplot(data, hue="class",diag_kind="hist")
    # sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
    # sns_plot.savefig("data_plot.png")
