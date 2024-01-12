import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

train_df = pd.read_csv("train_cleaned.csv")

test_df = pd.read_csv("test_cleaned.csv")

y_train = train_df["Exited"]
X = train_df.drop('Exited', axis=1)

x_test = test_df


# Standardize the features
scaler = StandardScaler()
X[:] = scaler.fit_transform(X)

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

class ChurnClassifier(nn.Module):
    
    def __init__(self, input_size, conv_channels=16):
        super(ChurnClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=3, padding = 1)
        self.fc2 = nn.Linear(conv_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for fully connected layer
        x = x.mean(dim=-1)  # Global average pooling
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
#%%
    
# Define loss function and optimizer
criterion = nn.BCELoss()
#weight_decay = 0.01

# Initialize the model
input_size = X.shape[1]
model = ChurnClassifier(input_size)

# Define the number of folds for cross-validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

y = y_train

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Initialize the model for each fold
    model = ChurnClassifier(input_size)

    # Initialize optimizer for each fold
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the test set for each fold
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test_binary = (y_pred_test > 0.5).float()

    # Calculate accuracy for each fold
    accuracy = accuracy_score(y_test, y_pred_test_binary.numpy())
    print(f'Accuracy on the test set (Fold {fold + 1}): {accuracy:.2%}')
    
    
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

# Standardize the features
scaler = StandardScaler()
x_test[:] = scaler.fit_transform(test_df.copy())

X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)


# Initialize the model for each fold
model = ChurnClassifier(input_size)

# Initialize optimizer for each fold
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

#%%

# Evaluate the model on the test set for each fold
with torch.no_grad():
    y_pred_test = (model(X_test_tensor) > 0.5).int()

test_df_result = pd.DataFrame(test_df['id'])

test_df_result['Exited'] = y_pred_test

test_df_result.to_csv("TwoLayer_NCN_Model.csv", index=False)


