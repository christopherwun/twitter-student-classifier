###############################################################################
''' Import Libraries '''
#preprocessing libraries
import pandas as pd
import emoji

#create emoji identifier
emojis = emoji.UNICODE_EMOJI.keys()
emojiString = ""
for e in emojis:
    emojiString += e

###############################################################################
''' Load + Preprocess Data '''
filepath = '' #insert filepath here
tweet_file = filepath + 'user_tweets.csv' #the raw tweets of the user
output_file = filepath + 'test_set.csv' #the labeled output of each user

tweet_df = pd.read_csv(tweet_file, index_col = 'Unnamed: 0', delimiter=',')
tweet_df = tweet_df.fillna('') #fill all empty columns (all cases where tweet count<200) with ''
output_df = pd.read_csv(output_file)['Output'].to_list()

###############################################################################
''' Create Tweet Features '''
feature_dict = {}
for user in tweet_df.index:
    tweets = tweet_df.loc[user,:]
    emojis = 0
    hashtag = 0
    hahalol = 0
    
    tweet_num = 200
    for tweet in tweets:
    ###########################################################################
    ''' Count Number of Tweets '''
        if tweet == '':
            tweet_num -= 1
        else:
            emoji_tf = False
            hashtag_tf = False
            hahalol_tf = False
    ###########################################################################
    ''' Determine of tweet contains each of 3 features '''
            for i in tweet:
                if i not in '"\'%-.,!?’@&$/#abcdefghijklmnopqrstuvwxyz1234567890|=+[]{}_;:></^*()~`' and i in emojiString:
                    emoji_tf = True
                if i == '#':
                    hashtag_tf = True
            if 'HAHA' in tweet or 'LOL' in tweet:
                hahalol_tf = True
    ###########################################################################
    ''' Total occurence counts '''
            if emoji_tf == True:
                emojis += 1
            if hashtag_tf == True:
                hashtag += 1
            if hahalol_tf == True:
                hahalol += 1
    ###########################################################################
    '''Average feature counts'''
    if tweet_num > 0:
        emojis /= tweet_num
        hashtag /= tweet_num
        hahalol /= tweet_num
    ###########################################################################
    '''Discretize all features'''
        if emojis == 0:
            emojis = '0.0'
        if hashtag == 0:
            hashtag = '0.0'
        if hahalol == 0:
            hahalol = '0.0'
        if emojis == 1:
            emojis = '0.9'
        if hashtag == 1:
            hashtag = '0.9'
        if hahalol == 1:
            hahalol = '0.9'
        feature_dict[user] = {'emojis':int(str(emojis)[2])+1, 'hashtag':int(str(hashtag)[2])+1, 'hahalol':int(str(hahalol)[2])+1}
    ###########################################################################
    ''' Mark Users with No Tweets '''
    if tweet_num == 0:
        print('x')
        feature_dict[user] = {'emojis':-1, 'hashtag':-1, 'hahalol':-1}
    
#Convert to dataframe
feature_df = pd.DataFrame.from_dict(feature_dict)
feature_df = feature_df.transpose()
#Attach outputs
feature_df['Output'] = output_df
feature_df.to_csv(filepath + 'tweet_features.csv')