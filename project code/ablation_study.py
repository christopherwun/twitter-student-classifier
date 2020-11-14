''' Import Libraries '''
#preprocessing libraries
from helper_functions import preprocess, csv_to_params
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

#Load and preprocess train and test sets (only the training sets are used in ablation study)
train_df = '../datasets/train_set.csv'
test_df = '../datasets/test_set.csv'

X_train, _, y_train, _ = preprocess(train_df, test_df, features)

#Format and fill the training set
X_train = pd.DataFrame(X_train)
X_train = X_train.fillna('')

###############################################################################
''' Ablation Study Function '''
def ablation_study(clf, clf_name):
    savepth = '' #insert savepath here
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    #create benchmark values (non-ablated)
    benchmark_dict = cross_validate(clf, X_train, y_train, cv=cv, scoring='f1')
    
    importance = []
    # iterate over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        ablated_score = cross_validate(clf, X_train.drop(col, axis=1), y_train, cv=cv, scoring='f1')
        importance.append(benchmark_dict['test_score'].mean() - ablated_score['test_score'].mean())
        
    #saving importance values into dataframe
    importance_df = pd.concat([pd.DataFrame.from_dict(features),
                               pd.DataFrame.from_dict(importance)], axis=1)
    importance_df = pd.DataFrame.from_dict(importance_df)
    
    #saving to csvfile
    savename = savepth + clf_name + '_ablation_study_full.csv'
    importance_df.to_csv(savename, index=False, header=False)
    
    #return relative importance values
    return importance_df
    
###############################################################################
''' Ablation Study '''
filepath = '' #insert filepath here

#Logistic Regression
params = csv_to_params(filepath + 'LogisticRegression_params.csv')
log_clf = LogisticRegression(random_state = 0, **params)
ablation_study(log_clf, 'LogisticRegression')

#Random Forest
params = csv_to_params(filepath + 'RandomForest_params.csv')
rf_clf = RandomForestClassifier(random_state = 0, **params)
ablation_study(rf_clf, 'RandomForest')

#Linear SVC
params = csv_to_params(filepath + 'LinearSVC_params.csv')
svm_clf = LinearSVC(dual=False, random_state = 0, **params)
ablation_study(svm_clf, 'LinearSVC')

#K-Nearest Neighbors
params = csv_to_params(filepath + 'KNeighbors_params.csv')
knn_clf = KNeighborsClassifier(**params)
ablation_study(knn_clf, 'KNeighbors')

#AdaBoost Classifier (does not utilize a params file because it requires an input model)
ada_clf = AdaBoostClassifier(random_state = 0, algorithm='SAMME', learning_rate=0.001, 
                             n_estimators = 50,base_estimator = rf_clf)
ablation_study(ada_clf, 'AdaBoost')

#Stacking Classifier (does not utilize a params file because it requires many input models)
estimators = [('tree',DecisionTreeClassifier(max_depth=1)), 
              ('svm',svm_clf), ('knn',knn_clf), ('rf',rf_clf), ('log',log_clf)]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=0))
ablation_study(stack_clf, 'StackedClass')