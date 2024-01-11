import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
x_train = train_df.drop('Exited', axis=1)

x_test = test_df

# Initialize the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier on the training data
nb_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(x_train)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_train, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_train, y_pred))

# Calculate AUC for estimating Kaggle Performance

auc_score = roc_auc_score(y_train, y_pred)
print(f'AUC: {auc_score:.2f}')

test_df_result = pd.DataFrame(test_df['id'])

test_pred = nb_classifier.predict(x_test)

test_df_result['Exited'] = test_pred

test_df_result.to_csv("Naive_Bayes_Model.csv", index=False)

