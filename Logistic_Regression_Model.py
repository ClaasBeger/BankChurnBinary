import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
x_train = train_df.drop('Exited', axis=1)

x_test = test_df

#%%

# Train a logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_train)

# Get predicted probabilities
y_prob = model.predict_proba(x_train)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_train, y_pred)
conf_matrix = confusion_matrix(y_train, y_pred)

# Display results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

#%%

# Compute AUC-ROC
auc_roc = roc_auc_score(y_train, y_prob)
print(f'AUC-ROC Score: {auc_roc}')