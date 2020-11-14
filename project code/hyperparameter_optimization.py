###############################################################################
''' Import Libraries '''
#preprocessing libraries
from helper_functions import preprocess
import pandas as pd
import numpy as np

#sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################
''' Load + Preprocess Data '''
#17-feature list:
features = ['Year?', 'NYC College?','Student?','Alum?','Occupation?',
            'Emojis','Parent?','Views My Own?',
            'Link?','Consecutive Upper','Name Emojis',
            'Name Title', 'Tweet Count','Followers',
            'Friends','Liked Posts','Verified']

#Load and preprocess train and test sets (only the training sets are used in hyperparameter optimization)
train_df = '../datasets/train_set.csv'
test_df = '../datasets/test_set.csv'

X_train, _, y_train, _ = preprocess(train_df, test_df, features)

#Format and fill the training set
X_train = pd.DataFrame(X_train)
X_train = X_train.fillna('')

###############################################################################
''' Metrics + Optimization Function '''
def save_metrics(clf, savename, splits):
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    scoring = ['accuracy', 'precision', 'recall', 'f1','roc_auc']
    scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=cv)
    
    results = {}
    for key in scores.keys():
        if 'test' in key:
            results[key] = np.mean(scores[key])
    
    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.to_csv('../data/SIML Models/' + savename + '_metrics.csv', header=False)
    print(result_df)

def optimized(clf, savename, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                              cv = cv, scoring = 'f1', verbose=2, n_jobs = -1)
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
#    params_df = pd.DataFrame.from_dict(best_params, orient='index')
#    params_df.to_csv('../data/SIML Models/' + savename + '_params.csv', header=False)
    return(best_clf)

################################################################################
''' Logistic Regression '''
#hyperparameter ranges
penalty = ['l1', 'l2']
tol = [float(x) for x in np.logspace(start = -10,stop = 5, num = 16)]
C = [float(x) for x in np.logspace(start= -10, stop=5, num= 16)]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
param_grid = {'penalty': penalty,'tol': tol,'C': C,'solver': solver}

#optimize + test
log_clf = optimized(LogisticRegression(random_state=0), 'LogisticRegression', param_grid)
save_metrics(log_clf, 'LogisticRegression', 10)

################################################################################
''' Random Forest '''
#hyperparameter ranges
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
min_samples_split = [2,7,10]
min_samples_leaf = [1,2,3,4]
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

#optimize + test
rf_clf = optimized(RandomForestClassifier(random_state=0), 'RandomForest', param_grid)
save_metrics(rf_clf, 'RandomForest', 10)

################################################################################
''' Linear SVM '''
#hyperparameter ranges
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
tol = [float(x) for x in np.logspace(start = -10,stop = 5, num = 16)]
C = [float(x) for x in np.logspace(start= -10, stop=5, num = 16)]
param_grid = {'kernel': kernel, 'tol':tol, 'C':C}

#optimize + test
svm_clf = optimized(SVC(random_state=0), 'SVM', param_grid)
save_metrics(svm_clf, 'SVM', 10)

################################################################################
''' K-Nearest Neighbors '''
#hyperparameter ranges
leaf_size = list(range(1,50,2))
n_neighbors = list(range(1,30,2))
p=[1,2]
param_grid = {'leaf_size': leaf_size, 'n_neighbors': n_neighbors, 'p': p}

#optimize + test
knn_clf = optimized(KNeighborsClassifier(), 'KNeighbors', param_grid)
save_metrics(knn_clf, 'KNeighbors', 10)

################################################################################
''' AdaBoost Classifier '''
#hyperparameter ranges
base_estimator = [DecisionTreeClassifier(max_depth=1), svm_clf, knn_clf, rf_clf, log_clf]
n_estimators = [int(x) for x in np.linspace(50,500,num=10)]
learning_rate = [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
algorithm = ['SAMME', 'SAMME.R']
param_grid = {'base_estimator': base_estimator,
              'n_estimators': n_estimators,
              'learning_rate': learning_rate,
              'algorithm': algorithm}

#optimize + test
ada_clf = optimized(AdaBoostClassifier(random_state=0), 'AdaBoost', param_grid)
save_metrics(ada_clf, 'AdaBoost', 10)