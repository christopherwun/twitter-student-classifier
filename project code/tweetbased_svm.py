###############################################################################
''' Import Libraries '''
import pandas as pd
import numpy as np
#sklearn libraries
from sklearn.svm import SVC
from helper_functions import evaluate_svm

###############################################################################
''' Load + Preprocess Data '''
#load datasets
train_df = pd.read_csv('train_tweet_features')
test_df = pd.read_csv('test_tweet_features.csv')

#drop users with zero tweets
train_df = train_df.drop(train_df.index[train_df['emojis'] == -1])
test_df = test_df.drop(test_df.index[test_df['emojis'] == -1])

#isolate features
X_train = train_df.drop(['Unnamed: 0','Output'],axis=1)
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
X_test = test_df.drop(['Unnamed: 0','Output'],axis=1)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

#isolate outputs
y_train = train_df['Output'].to_numpy(dtype=np.float64)
y_test = test_df['Output'].to_numpy(dtype=np.float64)

###############################################################################
''' Train + Evaluate Model '''
clf = SVC(kernel = 'linear')
clf.fit(X_train,y_train)

result = evaluate_svm(clf, X_test, y_test)