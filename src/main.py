import pandas as pd
import numpy as np
import os

"""Data Processing"""
#fnlwgt is weird, maybe look into adult.names for more info

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

data = pd.read_csv("../data/adult.data", header=None)
data.columns = columns
for c in columns:
    print(data[c])

