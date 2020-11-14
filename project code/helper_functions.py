''' Import Libraries '''
import pandas as pd
from sklearn import preprocessing
import emoji

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


