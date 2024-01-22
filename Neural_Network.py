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

#add_df = pd.read_csv("real_data_cleaned.csv")
#add_df_y = add_df["Exited"]
#add_df = add_df.drop(labels=['id', 'Exited'], axis=1)



y_train = train_df["Exited"]
#y_train = pd.concat([y_train, add_df_y])

x_test = test_df

reduced_train_df = train_df.drop(labels=['id', 'Surname_Origin_Longitude', 'Surname_Origin_Latitude'], axis = 1)

reduced_test_df = test_df.drop(labels=['id', 'Surname_Origin_Longitude', 'Surname_Origin_Latitude'], axis = 1)

X = reduced_train_df.drop('Exited', axis=1)
#X = pd.concat([X, add_df])

x_test = reduced_test_df

# Standardize the features
scaler = StandardScaler()
X[:] = scaler.fit_transform(X)

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

class ChurnClassifierConvWrapper(nn.Module):
    
    def __init__(self, input_size, conv_channels=16, norm='Layer', dropout=0.2):
        super(ChurnClassifierConvWrapper, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        #self.dropout = nn.Dropout(dropout)
# =============================================================================
#         if(norm=='Layer'):
#             self.norm1 = nn.LayerNorm(32)
#             self.norm2 = nn.LayerNorm(conv_channels)
#             self.norm3 = nn.LayerNorm(input_size)
#         else:
#             self.norm1 = nn.BatchNorm1d(32)
#             self.norm2 = nn.BatchNorm1d(conv_channels)
#             self.norm3 = nn.BatchNorm1d(input_size)
# =============================================================================
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=3, padding = 1)
        self.fc2 = nn.Linear(conv_channels, input_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.norm1(x)
        #x = self.dropout(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for fully connected layer
        x = x.mean(dim=-1)  # Global average pooling
        #x = self.norm2(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        #x = self.norm3(x)
        #x = self.dropout(x)
        return x
    
#%%

class ChurnClassifier2HBlock(nn.Module):
    
    def __init__(self, input_size, dropout=0.2, norm='Layer'):
        super(ChurnClassifier2HBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.norm = norm
        self.fc1 = nn.Linear(input_size, 32)
        if(norm == 'Layer'):
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(24)
            self.norm3 = nn.LayerNorm(input_size)
        else:
            self.norm1 = nn.BatchNorm1d(num_features=32)
            self.norm2 = nn.BatchNorm1d(num_features=24)
            self.norm3 = nn.BatchNorm1d(num_features=input_size)
        self.h1 = nn.Linear(32, 24)
        self.h2 = nn.Linear(24, input_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.h1.weight)
        nn.init.xavier_uniform_(self.h2.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.dropout(x)
        return x
    
class ChurnClassifier2H(nn.Module):
    
    def __init__(self, input_size, dropout=0.2, norm='Layer'):
        super().__init__()
        self.block1 = ChurnClassifier2HBlock(input_size, dropout, norm)
        self.block2 = ChurnClassifierConvWrapper(input_size, conv_channels=input_size, dropout=dropout, norm=norm)
        self.block3 = ChurnClassifier2HBlock(input_size, dropout, norm)
        self.block4 = ChurnClassifierConvWrapper(input_size, conv_channels=input_size, dropout=dropout, norm=norm)
        self.block5 = ChurnClassifier2HBlock(input_size, dropout, norm)
        self.fc2 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x = x + self.block4(x)
        x = x + self.block5(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
        
    
#%%
    
# Define loss function and optimizer
criterion = nn.BCELoss()
#weight_decay = 0.01

# Initialize the model
input_size = X.shape[1]
model = ChurnClassifier2H(input_size, dropout=0.2)

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
    model = ChurnClassifier2H(input_size, dropout=0.5)

    # Initialize optimizer for each fold
    base_lr = 0.001
    max_lr = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr,
                                                    max_lr=max_lr, step_size_up=200, cycle_momentum=False)
    

    # Training loop
    epochs = 1500
    for epoch in range(epochs):
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()
        
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_tensor)
                y_pred_test_binary = (y_pred_test > 0.5).float()
                accuracy = accuracy_score(y_test, y_pred_test_binary.numpy())
            model.train()
            print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.2%}')

    # Evaluate the model on the test set for each fold
    model.eval()
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

criterion = nn.BCELoss()

# Standardize the features
scaler = StandardScaler()
x_test[:] = scaler.fit_transform(x_test.copy())

X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)


# Initialize the model for each fold
model = ChurnClassifier2H(X.shape[1], dropout=0.2)

# Initialize optimizer for each fold
base_lr = 0.001
max_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=base_lr)
lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr,
                                                max_lr=max_lr, step_size_up=200, cycle_momentum=False)

# Training loop
epochs = 1000
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_schedule.step()

    if (epoch + 1) % 100 == 0:
        print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

#%%

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)

test_df_result = pd.DataFrame(test_df['id'])

test_df_result['Exited'] = y_pred_test

test_df_result.to_csv("NN_5B_Adam_CyclingLR_ConvNoNormDrops_Surnames_Excluded_probs.csv", index=False)

#%%

class ChurnClassifierBasic(nn.Module):
    
    def __init__(self, input_size):
        super(ChurnClassifierBasic, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.h1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
#%%
from sklearn.ensemble import RandomForestClassifier

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)

rf_model = RandomForestClassifier(n_estimators=150, min_samples_split=2, min_samples_leaf=4, max_depth=10)

rf_model.fit(X_train_tensor, y_train_tensor)

y_prob = rf_model.predict_proba(X_test_tensor)[:, 1]


ensemble_y_test = np.empty(shape=(110023))

for index in range(len(ensemble_y_test)):
    ensemble_y_test[index] = 0.5*y_pred_test.numpy()[index]+0.5*y_prob[index]
    
#ensemble_y_test = np.add(0.6*y_pred_test.numpy(),0.4*y_prob)

test_df_result = pd.DataFrame(test_df['id'])

test_df_result['Exited'] = ensemble_y_test

test_df_result.to_csv("NN_RF_Ensemble_probs_0.5.csv", index=False)


# %%

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)

# Create an XGBoost classifier
gbmodel = XGBClassifier(colsample_bytree=0.8, learning_rate=0.15, max_depth=3, n_estimators=100, 
                      subsample = 0.9)

# Train the model on the training data
gbmodel.fit(X_train_tensor, y_train_tensor)

# Make predictions on the testing data
y_pred = gbmodel.predict(X_test_tensor)

rf_model = RandomForestClassifier(n_estimators=150, min_samples_split=2, min_samples_leaf=4, max_depth=10)

rf_model.fit(X_train_tensor, y_train_tensor)

y_prob = rf_model.predict_proba(X_test_tensor)[:, 1]


ensemble_y_test = np.empty(shape=(110023))

for index in range(len(ensemble_y_test)):
    ensemble_y_test[index] = 0.5*y_pred_test.numpy()[index]+0.3*y_prob[index]+0.2*y_pred[index]
    
test_df_result = pd.DataFrame(test_df['id'])

test_df_result['Exited'] = ensemble_y_test

test_df_result.to_csv("NN_RF_XGB_Ensemble_probs.csv", index=False)





