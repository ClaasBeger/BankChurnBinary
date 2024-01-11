from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
from sklearn.model_selection import cross_val_predict

rf_model = RandomForestClassifier(n_estimators=100, random_state=77, max_depth=7)

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
x_train = train_df.drop('Exited', axis=1)

x_test = test_df

y_probas = cross_val_predict(rf_model, x_train, y_train, cv=5, method='predict_proba')

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

#print(classification_report(y_train, train_pred))
def save_model(model):
    
    train_df = pd.read_csv("train_cleaned.csv")

    test_df = pd.read_csv("test_cleaned.csv")

    y_train = train_df["Exited"]
    x_train = train_df.drop('Exited', axis=1)

    x_test = test_df

    model.fit(x_train, y_train)
    
    test_pred = model.predict(x_test)
    
    test_df_result = pd.DataFrame(test_df['id'])
    
    test_df_result['Exited'] = test_pred
    
    test_df_result.to_csv("Random_Forest_Model.csv", index=False)

#%%

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Use GridSearchCV with AUC as the scoring metric and 5-fold cross-validation
grid_search = GridSearchCV(rf_model, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(x_train, y_train)

# =============================================================================
# Best Hyperparameters: {'max_depth': 10, 
#                        'min_samples_leaf': 4, 
#                        'min_samples_split': 2, 
#                        'n_estimators': 150}
# 
# =============================================================================
# Print the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

#%%

rf_model = RandomForestClassifier(n_estimators=150, min_samples_split=2, min_samples_leaf=4, max_depth=10)

#save_model(rf_model)

#%%

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
x_train = train_df.drop('Exited', axis=1)

rf_model.fit(x_train, y_train)

y_prob = rf_model.predict_proba(x_train)[:, 1]

auc_roc = roc_auc_score(y_train, y_prob)
print(f'AUC-ROC Score: {auc_roc}')

#%%

## ENSEMBLE MODEL -------------------------------------------------------------

# Train a logistic regression model
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

# Initialize the Gaussian Naive Bayes classifier
model_nb = GaussianNB()

# Fit the classifier on the training data
model_nb.fit(x_train, y_train)

# Get predictions from each model on the validation set
pred_rf = rf_model.predict_proba(x_test)[:, 1]
pred_nb = model_nb.predict_proba(x_test)[:, 1]
pred_lr = model_lr.predict_proba(x_test)[:, 1]

# Simple averaging of predictions
ensemble_pred = (pred_rf*0.45 + pred_nb*0.35 + pred_lr*0.2)

# Evaluate the ensemble on the validation set
#ensemble_auc = roc_auc_score(y_train, ensemble_pred)
#print(f'Ensemble AUC: {ensemble_auc}')

test_df_result = pd.DataFrame(test_df['id'])

test_df_result['Exited'] = ensemble_pred>0.5

test_df_result.to_csv("Ensemble_Model.csv", index=False)

