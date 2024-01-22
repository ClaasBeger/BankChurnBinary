from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
x_train = train_df.drop('Exited', axis=1)

x_test = test_df

# Create an XGBoost classifier
model = XGBClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_train)

# Evaluate the model accuracy
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

y_probas = cross_val_predict(model, x_train, y_train, cv=5, method='predict_proba')

#rf_model.fit(x_train, y_train)

#train_pred = rf_model.predict(x_train)

# Optionally, you can calculate the mean AUC across all folds
# Calculate AUC for each class and each fold
for class_label in [0, 1]:
    auc = roc_auc_score((y_train == class_label).astype(int), y_probas[:, class_label])
    print(f"AUC for Class {class_label}: {auc}")

# Optionally, you can calculate the mean AUC across all folds for each class
mean_auc_class_0 = roc_auc_score(y_train == 0, y_probas[:, 0])
mean_auc_class_1 = roc_auc_score(y_train == 1, y_probas[:, 1])

print(f"Mean AUC for Class 0: {mean_auc_class_0}")
print(f"Mean AUC for Class 1: {mean_auc_class_1}")

#%%

model = XGBClassifier()

# Define a parameter grid for GridSearch
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create a GridSearchCV object with AUC as the scoring metric
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)

# Fit the GridSearchCV object to the data
grid_search.fit(x_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred_proba = best_model.predict_proba(x_test)[:, 1]

# Evaluate the model AUC
auc = roc_auc_score(y_train, y_pred_proba)
print(f"Best Parameters: {best_params}")
print(f"AUC with Best Parameters: {auc:.4f}")

#%%

# Refine the grid

model = XGBClassifier()

# Define a parameter grid for GridSearch
param_grid = {
    'learning_rate': [0.15, 0.1, 0.05],
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Create a GridSearchCV object with AUC as the scoring metric
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)

# Fit the GridSearchCV object to the data
grid_search.fit(x_train, y_train)

# Get the best parameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the testing data using the best model
y_pred_proba = best_model.predict_proba(x_train)[:, 1]

# Evaluate the model AUC
auc = roc_auc_score(y_train, y_pred_proba)
print(f"Best Parameters: {best_params}")
print(f"AUC with Best Parameters: {auc:.4f}")