import numpy as np
import pandas as pd
import os
from sklearn import preprocessing


train_transaction = pd.read_csv('./input/train_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('./input/train_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('./input/sample_submission.csv', index_col='TransactionID')

test_transaction = pd.read_csv('./input/test_transaction.csv', index_col='TransactionID')
test_identity = pd.read_csv('./input/test_identity.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

y_train = train['isFraud']
X_train = train.drop('isFraud', axis=1)
# print(X_train.info)

#train
temp_list_cat = list(X_train.select_dtypes('object').columns.values)
temp_list_int = list(X_train.select_dtypes('int64').columns.values)
temp_list_float = list(X_train.select_dtypes('float64').columns.values)
temp_list_zhuan = temp_list_cat+temp_list_int

train_cat_int = X_train[temp_list_zhuan]
train_float = X_train[temp_list_float].fillna(-999)
#
# #test
# temp_list_cat = list(test.select_dtypes('object').columns.values)
# temp_list_int = list(test.select_dtypes('int64').columns.values)
# temp_list_float = list(test.select_dtypes('float64').columns.values)
# temp_list_zhuan = temp_list_cat+temp_list_int

test_cat_int = test[temp_list_zhuan]
test_float = test[temp_list_float].fillna(-999)

del train, train_transaction, train_identity, test, test_transaction, test_identity,X_train

train_cat_int = train_cat_int.fillna(-999)
test_cat_int = test_cat_int.fillna(-999)
# train_cat_int = train_cat_int.astype('object')
# test_cat_int = test_cat_int.astype('object')

# X_train = X_train.fillna(-999)
#
# X_test = test.copy()
# X_test = X_test.fillna(-999)
#
train_cat_int = train_cat_int.applymap(str)
test_cat_int = test_cat_int.applymap(str)
#
# del train, train_transaction, train_identity, test, test_transaction, test_identity
for f in train_cat_int.columns:
    if train_cat_int[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_cat_int[f].values) + list(test_cat_int[f].values))
        train_cat_int[f] = lbl.transform(list(train_cat_int[f].values))
        test_cat_int[f] = lbl.transform(list(test_cat_int[f].values))

# for temp_a in temp_list_float:
#   temp = train_float[temp_a]
#   temp_normal = (temp - temp.mean()) / (temp.std())
#   train_float[temp_a] = temp_normal
#
# for temp_a in temp_list_float:
#   temp = test_float[temp_a]
#   temp_normal = (temp - temp.mean()) / (temp.std())
#   test_float[temp_a] = temp_normal


# def check_cunique(df, cols):
#  """check unique values for each column
#  df: data frame.
#  cols: list. The columns of data frame to be counted
#  """
#  df_nunique = df[cols].nunique().to_frame()
#  df_nunique = df_nunique.reset_index().rename(columns={'index': 'feat', 0: 'nunique'})
#  return df_nunique

# df_nunique = check_cunique(train_cat_int,cols=train_cat_int.columns.values)
# print(df_nunique)
#
# df_nunique = check_cunique(test_cat_int,cols=test_cat_int.columns.values)
# print(df_nunique)

final_train_int = train_cat_int.values
final_train_float = train_float.values
final_test_int = test_cat_int.values
final_test_float = test_float.values
target = y_train.values

del train_cat_int,train_float,y_train,test_cat_int,test_float



# df_nunique = check_cunique(final_train_int,cols=final_train_int.columns.values)
# final_train = X_train.values
# final_test = X_test.values
# target = y_train.values
# del X_train,X_test,y_train
field_dims = np.max(final_train_int,axis=0)+1
# field_dims2 = np.max(final_test_int,axis=0)+1

# print(field_dims1)
# print(field_dims2)

