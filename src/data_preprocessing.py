import pandas as pd
import numpy as np
import os
from itertools import product

"""Data Processing"""
# Removed the first line "|1x3 Cross validator" in adult.test, because pd.read_csv failed and it was useless

# 15 columns
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

if __name__ == "__main__":
    for df_name in ["data", "test"]:
        df = pd.read_csv("../data/adult." + df_name, names=columns, header=None)
        # rows in df: train=32561, test=16281
        indices_to_remove = [i for i, c in product(df.index, df.columns) if str(df.at[i, c]).__contains__("?")]
        df = df.drop(indices_to_remove)
        # rows in df: train=30162, test=15060
        df.to_csv("../data_processed/" + df_name + ".csv",index=None)
