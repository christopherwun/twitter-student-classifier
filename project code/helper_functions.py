''' Import Libraries '''
import pandas as pd
from sklearn import preprocessing, metrics
import emoji
from sklearn.metrics import plot_roc_curve, roc_auc_score

###############################################################################
''' Set Up Emoji Identifier '''
emojis = emoji.UNICODE_EMOJI.keys()
emojiString = ""
for e in emojis:
    emojiString += e

###############################################################################
''' Preprocess Training and Test Sets '''
def preprocess(train_csv, test_csv, features):      
    train_df = pd.read_csv(train_csv, sep=',')
    test_df = pd.read_csv(test_csv, sep=',')
    
    #separate train + test data
    y_train = train_df['Output']
    y_test = test_df['Output']

    #polynomialize
    train_df['Emojis'], test_df['Emojis'] = (train_df['Emojis'])**2, (test_df['Emojis'])**2
    
    #isolate the desired features
    train_df = train_df.loc[:,features]
    test_df = test_df.loc[:,features]
    
    train_df = train_df.astype(float)
    
    train_df = train_df.fillna('')
    test_df = test_df.fillna('')
    
    #scale features
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_df)
    
    X_train = scaler.transform(train_df)
    X_test = scaler.transform(test_df)

    results = [X_train, X_test, y_train, y_test]
    return results
    
###############################################################################
''' Convert CSV files of parameters into usable dictionaries '''
def csv_to_params(filename):
    #read parameters files into a variable
    paramdf = pd.read_csv(filename, header=None, index_col=0)    
    params = paramdf.to_dict()[1]
    
    #return a correctly formatted dictionary for use in sklearn
    for key in params:
        try:
            params[key] = float(params[key])
        except:
            pass
        if key in ['n_neighbors','n_estimators','min_samples_leaf','min_samples_split']:
            params[key] = int(params[key])
    
    return params
    
###############################################################################
''' Evaluate model performance by getting test metrics '''
def evaluate(model, test_input, test_output, **kwargs):
    # threshold is the probability above which a user will be classified as a student
    threshold = kwargs.get('threshold', 0.5)
    verbose = kwargs.get('verbose',0)
    # bounds parameter is used to evaluate gray area models
    bounds = kwargs.get('bounds',[threshold,threshold])
    
    test_output = test_output.reset_index(drop=True)
    
    # get the auc of predictions
    preds = model.predict_proba(test_input)
    preds = preds[:,1]
    auc = metrics.roc_auc_score(test_output, preds)
    
    # convert prediction probabilities into labels of "student" "non-student" or "uncertain"
    classifs = []
    for i in range(len(preds)):
        if preds[i]>bounds[0] and preds[i]<bounds[1]:
            classifs.append(-1)
        elif preds[i]>bounds[1]:
            classifs.append(1)
        else:
            classifs.append(0)
    
    # compare predicted label to true value, record true positives, true negatives, false positives, false negatives
    preds = model.predict(test_input)
    tp, tn, fp, fn, unc = 0, 0, 0, 0, 0
    
    for k in range(len(preds)):
        p = classifs[k]
        t = int(test_output[k])
        
        if(p<0):
            unc += 1
        elif(p>=0.5):
            if(t>=0.5):
                tp += 1
            else:
                fp += 1
        else:
            if(t<0.5):
                tn += 1
            else:
                fn += 1
    
    # compute test metrics: accuracy, precision, recall, f1, and coverage
    accuracy= (tp+tn) / (tp+tn+fn+fp)
    precision = 0
    if(tp!=0):
        precision = (tp) / (tp + fp)
    recall = 0
    if(tp!=0):
        recall = (tp) / (tp + fn)  
    f1 = 0
    if(precision!=0 or recall !=0):
        f1 = 2*precision*recall/(precision+recall)
    coverage = 1 - (unc/(fp+fn+tp+tn+unc))
    metrics_list = {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'unc':unc,'coverage':coverage,'roc_auc':auc}
    
    if verbose==1:
        print("Accuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}\nTP,TN,FP,FN, UNC: {},{},{},{},{}".format(accuracy,precision,recall,f1,tp,tn,fp,fn,unc))
    
    return metrics_list

###############################################################################
''' Evaluate SVM models, which do not give prediction probabilities '''
def evaluate_svm(model, test_input, test_output):
    classifs = model.predict(test_input)
    
    # get auc  values
    roc = plot_roc_curve(model, test_input, test_output, name='ROC fold {}')
    auc = roc.roc_auc
    
    # compare predicted label to true value, record true positives, true negatives, false positives, false negatives
    tp, tn, fp, fn, unc = 0, 0, 0, 0, 0
    for k in range(len(classifs)):
        p = classifs[k]
        t = int(test_output[k])
        
        if(p<0):
            unc += 1
        elif(p>=0.5):
            if(t>=0.5):
                tp += 1
            else:
                fp += 1
        else:
            if(t<0.5):
                tn += 1
            else:
                fn += 1
    
    # compute test metrics: accuracy, precision, recall, f1, and coverage
    accuracy= (tp+tn) / (tp+tn+fn+fp)
    precision = 0
    if(tp!=0):
        precision = (tp) / (tp + fp)
    recall = 0
    if(tp!=0):
        recall = (tp) / (tp + fn)
    f1 = 0
    if(precision!=0 or recall !=0):
        f1 = 2*precision*recall/(precision+recall)
        
    metrics_list = {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,'roc_auc':auc}
    return metrics_list