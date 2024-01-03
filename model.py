import pandas as pd

import torch

import numpy as np

train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

#%%

# Small data description (can be deleted later):
    
# =============================================================================
# id - Unique ID. We can use this features as a count for other parameters.
# 
# CustomerID - Customer ID is not a unique ID, there are few customer IDs which have more than 80 repetitive occurrences.
# 
# Surname - This is the surname of the customer, there are too many repititions. Actually there are only around 2700 unique surnames.
# 
# Credit Score - Your creditworthiness is rated by a three-digit figure called a credit score. 300 to 850 is the range of FICO scores.
# You have a better chance of getting approved for loans and better prices the higher your score. Now in our dataset there are range of values starting from 350 and going up to 850. Now this can be a very useful information while thinking about the churning.
# 
# Geography - There are 3 unique values - France, Spain, and Germany. One has to use Label Encoder or OneHotEncoder to encode these values.
# 
# Gender - There are only 2 unique values - Male and Female. Here a label binarizer is enough for the encoding purposes.
# 
# Age - Depicts the age of the customers. There are all possible values starting from 18 up to 92. 
# There are 2 anomalies found in the age column - there 2 values in float - 32.44 and 36.44. It would be better if we can round those values to 32 and 36 respectively.
# 
# Tenure - It might show from how many years the customer has been related to the bank or may be vice versa. There are values ranging from 0 to 10.
#  Most probably these values are in years.
# 
# Balance - This is the bank balance of the customer. There were many doubts in the discussion forum that the bank balance was 0. 
# When I performed the analysis, I found that actually 89000+ people had 0 bank balance. While the maximum amount recorded was around 250,000.
# 
# Number of Products - Now this can be a very difficult question. While there are only 4 unique values possible - 1, 2, 3, and 4.
#  This can be attributes to how many major/big products the customer owns. Or other explanation might be that how many products the customer has bought on loan.
# 
# Has Credit Card - Clear cut, whether the customer has a credit card or not. Same goes for the next column as well Is Active Member. 
# 
# Estimated Salary - What is the estimated salary of the individual. Now, this is a very important aspect of the real life scenario. Whenever you are given a credit from the bank,
# they mostly ask for whether or not you are salaried. If you are estimated of getting a higher salary, easier for them to credit you a higher amount of loan.
# =============================================================================

#%%

# Look for nan values and categorical variables

train_df.info()

# No null values

print(train_df.applymap(np.isreal).all(0))

# Identified Three Categorical Variables: Surname, Geography and Gender

#%%

train_df.head()

#%%

#Inspect different attributes

train_df.id.describe()

# ID is unique, seems to be relating to process (distinct applicant list also possible)

#%%

len(train_df.CustomerId.unique())

# 23221 distinct values -> not unique (reuse possible, or multiple processes per Customer)

# To investigate, we look at rows with same CustomerId

train_df['CustomerId'].duplicated()

#df = train_df[train_df.duplicated('CustomerId', keep=False)].sort_values('CustomerId')

# Customer ID does not display any obvious pattern or relation to the respective process

# I will replace the CustomerID with the respective count

CustomerIdCounts = train_df['CustomerId'].value_counts()

# Create a new column with counts based on CustomerId
train_df['CustomerIdCounts'] = train_df['CustomerId'].map(CustomerIdCounts)

# Drop the old 'CustomerId' column if needed
train_df.drop(columns=['CustomerId'], inplace=True)

# Rename the new column to 'CustomerId'
train_df.rename(columns={'CustomerIdCounts': 'CustomerId'}, inplace=True)

# Potentially use a zscaler to transform

from sklearn import preprocessing

# create a scaler using Z score normalization / Standardization on the Fare attribute
# The Fare varies greatly, thus we want to normalize this attribute
zscore_scaler = preprocessing.StandardScaler().fit(train_df[['CustomerId']])

#Apply the Zscore scaler Fare attribute and assign to a new Zscore column
train_df['CustomerId_zscore']=zscore_scaler.transform(train_df[['CustomerId']])

train_df['CustomerId_zscore'].describe()


#%%

# One-Hot encoding of gender

#train_df = pd.get_dummies(train_df, columns=['Gender'], drop_first=True)

