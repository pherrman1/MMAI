import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer



def get_data(full_data = True, features_to_drop = []):
    # Load data
    train_data = pd.read_csv("../data_processed/data.csv")
    test_data = pd.read_csv("../data_processed/test.csv")

    # Set labels to class, change string to 0 and 1
    train_labels = train_data["class"]
    train_labels.replace([" <=50K", " >50K"], [0, 1], inplace=True)
    test_labels = test_data["class"]
    test_labels.replace([" <=50K.", " >50K."], [0, 1], inplace=True)

    #save fnlwgt
    fnlwgt = dict(train_data["fnlwgt"])

    # Combine train and test for easier processing, label them for splitting later
    train_data["train"] = 1
    test_data["train"] = 0
    data = pd.concat([train_data, test_data], ignore_index=True)

    #Drop "education", since we have "education-num"
    data.drop("education", axis=1, inplace=True)

    #Drop certain features in features_to_drop
    if not full_data:
        data.drop(features_to_drop, axis=1, inplace=True)

    #OneHotEncoding for categorical columns except for label column
    input_data = data.loc[:, data.columns != 'class']

    # recreate train and test input sets
    #input_data.drop("remainder__fnlwgt", axis=1, inplace=True)
    train_input = input_data[input_data["train"] == 1]
    test_input = input_data[input_data["train"] == 0]
    train_input = train_input.drop("train", axis=1)
    test_input = test_input.drop("train", axis=1)

    return train_input, train_labels, test_input, test_labels, fnlwgt, data


if __name__ == "__main__":
    """Data Processing"""
    # Removed the first line "|1x3 Cross validator" in adult.test, because pd.read_csv failed and it was useless

    # 15 columns
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship",
               "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

    for df_name in ["data", "test"]:
        df = pd.read_csv("../data/adult." + df_name, names=columns, header=None)
        # rows in df: train=32561, test=16281
        indices_to_remove = [i for i, c in product(df.index, df.columns) if str(df.at[i, c]).__contains__("?")]
        df = df.drop(indices_to_remove)
        # rows in df: train=30162, test=15060
        df.to_csv(f"../data_processed/{df_name}.csv",index=None)

        print((df["capital-gain"][df["workclass"] == " Self-emp-not-inc"]))
