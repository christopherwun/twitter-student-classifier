###############################################################################
''' Import Libraries '''
from SIHelperMethods import preprocess, evaluate, evaluate_svm, csv_to_params
import pandas as pd
import pickle

#sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################
''' Load + Preprocess Data '''
#17-feature list:
features = ['Year?', 'NYC College?','Student?','Alum?','Occupation?',
            'Emojis','Parent?','Views My Own?',
            'Link?','Consecutive Upper','Name Emojis',
            'Name Title', 'Tweet Count','Followers',
            'Friends','Liked Posts','Verified']

#Load and preprocess train and test sets (only the training sets are used in ablation study)
train_df = '../datasets/train_set.csv'
test_df = '../datasets/test_set.csv'

X_train, X_test, y_train, y_test = preprocess(train_df, test_df, features)

#Format and fill the training set
X_train = pd.DataFrame(X_train)
X_train = X_train.fillna('')
    
###############################################################################
''' Train Optimized Models '''
filepath = '' #insert filepath

# Logistic Regression
params = csv_to_params(filepath + 'LogisticRegression_params.csv')
log_clf = LogisticRegression(random_state = 0, **params)
log_clf.fit(X_train, y_train)

#Random Forest
params = csv_to_params(filepath + 'RandomForest_params.csv')
rf_clf = RandomForestClassifier(random_state = 0, **params)
rf_clf.fit(X_train, y_train)

#Linear SVC
params = csv_to_params(filepath + 'LinearSVC_params.csv')
svm_clf = LinearSVC(dual=False, random_state = 0, **params)
svm_clf.fit(X_train, y_train)

#K-Nearest Neighbors
params = csv_to_params(filepath + 'KNeighbors_params.csv')
knn_clf = KNeighborsClassifier(**params)
knn_clf.fit(X_train, y_train)

#AdaBoost Classifier
ada_clf = AdaBoostClassifier(random_state = 0, base_estimator = RandomForestClassifier(max_depth=30, n_estimators=400, random_state=0), learning_rate=0.001, algorithm='SAMME', n_estimators=50)
ada_clf.fit(X_train, y_train)

#Stacking Classifier
estimators = [('tree',DecisionTreeClassifier(max_depth=1)), 
              ('svm',svm_clf), ('knn',knn_clf), ('rf',rf_clf), ('log',log_clf)]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_clf.fit(X_train, y_train)

###############################################################################
''' Save Model and Metrics Function '''
def save_model(clf, metrics, savepath):
    pickle.dump(clf, open(savepath + '_model.pkl', 'wb'))
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(savepath + '_metrics_final.csv')

###############################################################################
''' Get Test Metrics '''
log_metrics = evaluate(log_clf, X_test, y_test)
# save_model(log_clf, log_metrics, filepath + 'LogisticRegression')

rf_metrics = evaluate(rf_clf, X_test, y_test)
# save_model(rf_clf, rf_metrics, filepath + 'RandomForest')

svm_metrics = evaluate_svm(svm_clf, X_test, y_test)
# save_model(svm_clf, svm_metrics, filepath + 'LinearSVC')

knn_metrics = evaluate(knn_clf, X_test, y_test) 
# save_model(knn_clf, knn_metrics, filepath + 'KNearestNeighbors')

ada_metrics = evaluate(ada_clf, X_test, y_test)
# save_model(ada_clf, ada_metrics, filepath + 'AdaBoost')

stack_metrics = evaluate(stack_clf, X_test, y_test) 
# save_model(stack_clf, stack_metrics, filepath + '../StackedClass_final')
