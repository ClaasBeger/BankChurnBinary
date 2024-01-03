import pandas as pd

import torch

import numpy as np

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

#%%

# Look for nan values and categorical variables

train_df.info()

# No null values

print(train_df.applymap(np.isreal).all(0))

# Identified Three Categorical Variables: Surname, Geography and Gender

#%% Drop obsolete id column

train_df.drop(labels='id', axis=1, inplace=True)

#%%

# One-Hot encoding of gender

train_df = pd.get_dummies(train_df, columns=['Gender'], drop_first=True)