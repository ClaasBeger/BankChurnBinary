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

train_df.drop('CustomerId', axis=1, inplace=True)

#%%

# One-Hot encoding of gender

train_df = pd.get_dummies(train_df, columns=['Gender'], drop_first=True)

#%%

# Classify names by their origin using corresponding lookup table

from names_dataset import NameDataset, NameWrapper

nd = NameDataset()

#%%

# Add new column with Country of surname origin

def lookUpOrigin(row):
    entry = nd.search(row['Surname'])
    if(entry['last_name'] == None):
        return 'Unknown'
    return next(iter(nd.search(row['Surname'])['last_name']['country'].keys()))

train_df['Name_Origin'] = train_df.apply(lookUpOrigin, axis=1)

# Next I would like to add a column that encodes the country origin information
# through its location data 

from geopy.geocoders import Nominatim
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

# List of countries for name origins which we want to retrieve coordinates for
countries = train_df['Name_Origin'].unique()

# Initialize geolocator
geolocator = Nominatim(user_agent="country_locator")

coordinates = {}

# Retrieve coordinates for each country
for country in countries:
    location = geolocator.geocode(country)
    if location:
        coordinates[country] = (location.latitude, location.longitude)
    else:
        print(f"Coordinates not found for {country}")

# Create DataFrame from coordinates
#coordinate_df = pd.DataFrame(coordinates.values(), index=coordinates.keys(), columns=['Latitude', 'Longitude'])

# Calculate distances between countries
# =============================================================================
# distances = {}
# for country1, coord1 in coordinates.items():
#     for country2, coord2 in coordinates.items():
#         if country1 != country2 and (country2, country1) not in distances:
#             dist = geodesic(coord1, coord2).kilometers
#             distances[(country1, country2)] = dist
# =============================================================================

# Create a distance matrix
# =============================================================================
# distance_matrix = pd.DataFrame(0, index=coordinate_df.index, columns=coordinate_df.index)
# for (country1, country2), dist in distances.items():
#     distance_matrix.loc[country1, country2] = dist
#     distance_matrix.loc[country2, country1] = dist
# =============================================================================

# Apply Multidimensional Scaling (MDS) for dimensionality reduction
# =============================================================================
# mds = MDS(n_components=1, dissimilarity='precomputed', random_state=42)
# numeric_values = mds.fit_transform(distance_matrix)
# =============================================================================

# Scale the numeric values to a desired range
# =============================================================================
# scaler = MinMaxScaler()
# coordinate_df['NumericCountry'] = scaler.fit_transform(numeric_values)
# =============================================================================

# Replace value for unknown by average

tuples = [value for key, value in coordinates.items() if key != 'Unknown']

# Calculate the averages for each component of the tuples
avg_1 = sum(x[0] for x in tuples) / len(tuples)
avg_2 = sum(x[1] for x in tuples) / len(tuples)

# Update the 'Unknown' key with the averages
coordinates['Unknown'] = (avg_1, avg_2)

train_df['Surname_Origin_Latitude'] = train_df['Surname'].map(lambda x: coordinates.get(x, coordinates['Unknown'])[0])
train_df['Surname_Origin_Longitude'] = train_df['Surname'].map(lambda x: coordinates.get(x, coordinates['Unknown'])[1])


# NumericCountry does not reflect distance too well, instead I just insert Longitude and Latitute

train_df['Geography_Latitude'] = train_df['Geography'].map(lambda x: coordinates[x][0])
train_df['Geography_Longitude'] = train_df['Geography'].map(lambda x: coordinates[x][1])

#%%

# drop unneeded columns

train_df.drop(labels=['Surname', 'Geography', 'Name_Origin'], axis=1, inplace=True)

