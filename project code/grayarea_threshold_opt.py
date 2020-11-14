###############################################################################
''' Import Libraries '''
#preprocessing libraries
from SIHelperMethods import preprocess, evaluate, csv_to_params
import pandas as pd

#sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
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

#Load and preprocess train and test sets (only the training sets are used in grey area/threshold optimization)
train_df = '../datasets/train_set.csv'
test_df = '../datasets/test_set.csv'

X_train, _, y_train, _ = preprocess(train_df, test_df, features)

#Format and fill the training set
X_train = pd.DataFrame(X_train)
X_train = X_train.fillna('')

###############################################################################
''' Grey Area Optimization Function '''
def optimize_grey_area(clf, clf_name):    
    kfold = StratifiedKFold(n_splits=10)
    
    columns = ['accuracy','precision','recall','f1','tp','tn','fp','fn','unc','coverage','auc']

    bound_sizes = [.1,.2,.3,.4]
    lower_bounds = [i/100 for i in range(20, 80, 5)]
    bounds = []
    for size in bound_sizes:
        for low_bound in lower_bounds:    
            high_bound = round(low_bound + size, 2)
            if high_bound <= .9:
                bounds.append((low_bound, high_bound))
                
    t_results = pd.DataFrame(index=bounds, columns=columns)
    t_results = t_results.fillna(0)

    for train, test in kfold.split(X_train, y_train):
         for i in range(len(bounds)):
            fitted_clf = clf.fit(X_train[train], y_train[train])
            metric_list = evaluate(fitted_clf, X_train[test],y_train[test],bounds=bounds[i])
            
            #add all the metrics (will average after)
            t_results.loc[[bounds[i]]] += metric_list
    
    #take average
    t_results /= 10
    
    savepath = '../data/SIML Models/grey_area/'
    t_results.to_csv(savepath + clf_name + '_grey_area_opt.csv')
        
###############################################################################
''' Threshold Optimization Function '''
def optimize_threshold(clf, clf_name):
    kfold = StratifiedKFold(n_splits=10)
    
    columns = ['accuracy','precision','recall','f1','tp','tn','fp','fn','unc','coverage','auc']
    thresholds = [i/100 for i in range(10, 90, 5)]
    
    t_results = pd.DataFrame(index=thresholds, columns=columns)
    t_results = t_results.fillna(0)
    
    for train, test in kfold.split(X_train, y_train):
        for i in range(len(thresholds)):
            fitted_clf = clf.fit(X_train[train], y_train[train])
            metric_list = evaluate(fitted_clf, X_train[test],y_train[test],threshold=thresholds[i])
            
            #add all the metrics (will average after)
            t_results.loc[thresholds[i]] += metric_list
    
    #take averages
    t_results /= 10
    
    #save to csv
    savepath = '../data/SIML Models/prediction_thresholds/'
    t_results.to_csv(savepath + clf_name + '_threshold_opt.csv')

###############################################################################
''' Threshold + Grey Area Optimization on all Models '''
filepath = '' #insert filepath here

#Logistic Regression
params = csv_to_params(filepath + 'LogisticRegression_params.csv')
log_clf = LogisticRegression(random_state = 0, **params)
optimize_grey_area(log_clf, 'LogisticRegression')
optimize_threshold(log_clf, 'LogisticRegression')

#Random Forest
params = csv_to_params(filepath + 'RandomForest_params.csv')
rf_clf = RandomForestClassifier(random_state = 0, **params)
optimize_grey_area(rf_clf, 'RandomForest')
optimize_threshold(rf_clf, 'RandomForest')

#Linear SVC
params = csv_to_params(filepath + 'LinearSVC_params.csv')
svm_clf = LinearSVC(dual=False, random_state = 0, **params)
optimize_grey_area(svm_clf, 'LinearSVC')
optimize_threshold(svm_clf, 'LinearSVC')

#K-Nearest Neighbors
params = csv_to_params(filepath + 'KNeighbors_params.csv')
knn_clf = KNeighborsClassifier(**params)
optimize_grey_area(knn_clf, 'KNeighbors')
optimize_threshold(knn_clf, 'KNeighbors')

#AdaBoost Classifier
ada_clf = AdaBoostClassifier(random_state = 0, algorithm='SAMME', learning_rate=0.001, 
                             n_estimators = 50,base_estimator = rf_clf)
optimize_grey_area(ada_clf, 'AdaBoost')
optimize_threshold(rf_clf, 'AdaBoost')

#Stacking Classifier
estimators = [('tree',DecisionTreeClassifier(max_depth=1)), 
              ('svm',svm_clf), ('knn',knn_clf), ('rf',rf_clf), ('log',log_clf)]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
optimize_grey_area(stack_clf, 'StackedClass')
optimize_threshold(rf_clf, 'StackedClass')