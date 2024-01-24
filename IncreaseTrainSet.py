import pandas as pd

# Loading the data
df_train = pd.read_csv("train.csv", index_col='id')
df_test = pd.read_csv("test.csv", index_col='id')
df_churn_modelling = pd.read_csv("10k_Churn.csv", index_col='RowNumber')

#%%

#df_churn_modelling.rename(columns={"RowNumber": "id"}, inplace=True)


# Dropping unique identifiers and columns not present in df_test from df_train and df_churn_modelling
common_columns = df_test.columns.intersection(df_train.columns).intersection(df_churn_modelling.columns)
df_train_common = df_train[common_columns]
df_churn_modelling_common = df_churn_modelling[common_columns]

# Add/modify these lines
df_test_common = df_test[common_columns]
combined_df = pd.concat([df_train_common, df_test_common, df_churn_modelling_common], ignore_index=True)

# Counting the number of duplicated entries
duplicated_count = combined_df.duplicated().sum()


df = df_train.join(df_churn_modelling.set_index(list(df_test.columns)),on=list(df_test.columns),how='inner',rsuffix='_o')
print(F'{len(df)=}')
print(F'{(df["Exited"]^df["Exited_o"]).all()}')

# %%

# Merge the dataframes with indicator=True
merged_df = df_train.merge(df_churn_modelling, how='outer', on=list(df_test.columns), indicator=True)

# Filter out the overlapping samples
df_train = merged_df[merged_df['_merge'] != 'both'].drop('_merge', axis=1)

df_train['Exited_y'] = df_train['Exited_y'].apply(lambda x : x if x is None else abs(x-1))

# Combine Exited_x and Exited_y into a new column Exited
df_train['Exited'] = df_train['Exited_x'].combine_first(df_train['Exited_y'])

# Drop the original columns
df_train = df_train.drop(['Exited_x', 'Exited_y'], axis=1)

df_train.to_csv("Extended_train_uncleaned.csv", index=True, index_label='id')