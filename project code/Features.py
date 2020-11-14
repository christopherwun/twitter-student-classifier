###############################################################################
''' Import Libraries '''
from datetime import datetime
from dateutil.parser import parse
import emoji
import os    

###############################################################################
''' Define Emoji Count Function '''
def num_emojis(bio_text):
    count = 0
    for character in bio_text:
        if character in emoji.UNICODE_EMOJI:
            count += 1
    return count

###############################################################################
''' Load Occupations List '''
with open('Occupations.txt', 'r') as file:
    occupations = []
    for line in file:
        line = line[:-1]
        occupations.append(line.lower())

###############################################################################
''' Define all Features '''
class Features():
    def __init__(self, user):
        #######################################################################
        '''BASIC METADATA FEATURES''' #(statuses = tweets, favorites = likes by user):
        #Verified?
        self.verified = int(user['verified'])       #added 6/4
        #Link?
        self.link = 1                               #added 6/5
        if user['url'] == None:
            self.link = 0
        #Followers
        self.followers = user['followers_count']
        #Friends
        self.friends = user['friends_count']
        #Liked Posts
        self.liked_posts = user['favourites_count']
        #Tweet Count
        self.tweet_count = user['statuses_count']  #added 6/3, mod 6/10

        
        #######################################################################
        '''TIME-BASED FEATURES''' #(converting dates to numbers):                       
        #Created At
        created_str = str(parse(user['created_at'])) #added 6/4, mod 6/10
        self.created_inseconds = datetime.fromisoformat(created_str).timestamp() #**datetime.fromisoformat() requires date to be in YYYY-MM-DD HH:MM:SS.
        self.created_at = self.created_inseconds / 60 / 60 / 24 / 365.25         #unit: year
        #Last Tweet Time
        try:
            last_tweet_str = str(parse(user['status']['created_at']))
            self.last_tweet_inseconds = datetime.fromisoformat(last_tweet_str).timestamp()  #added 6/4
            self.last_tweet = self.last_tweet_inseconds / 60 / 60 / 24 / 365.25        #added 6/10
        except KeyError:
            self.last_tweet = None
        
        #######################################################################
        '''NAME-BASED FEATURES'''
        self.name = user['name']            #added 6/23
        #Consecutive Upper
        history = []
        count = 0
        for letter in user['name']:
            cap = letter.isupper()
            if cap == True:
                count += 1
            if cap == False: 
                history.append(count)
                count = 0
        try:
            self.cons_upper = max(history)
        except:
            self.cons_upper = 0             #added 6/23
        #Name Emojis
        self.name_emojis = num_emojis(self.name) 
        #Name Title
        self.name_prof = 0
        name_lower = self.name.lower()
        if ('mr.' in name_lower or 'mrs.' in name_lower or 
            'ms.' in name_lower or 'miss' in name_lower or
            'ph.d' in name_lower or 'ph. d' in name_lower or 'phd' in name_lower or
            'm.d' in name_lower or 'm. d' in name_lower or
            'doctor' in name_lower or 'dr.' in name_lower):
            self.name_prof = 1
        
        #######################################################################
        '''DESCRIPTION-BASED FEATURES'''
        self.desc = user['description'].lower()     #added 6/3   
        self.user_id = user['id']                   #added 6/3
        self.screen_name = user['screen_name']      #added 6/3
        #Year?
        self.year= 0                                #added 6/3
        if ("'2" in self.desc or "20'" in self.desc or
            "21'" in self.desc or "22'" in self.desc or
            "23'" in self.desc or "24'" in self.desc or
            "25'" in self.desc or "class of 202" in self.desc or
            "‘2" in self.desc or "20‘" in self.desc or 
            "21‘" in self.desc or "22‘" in self.desc or #apostrophe added 6/24
            "23‘" in self.desc or "24‘" in self.desc or 
            "25‘" in self.desc or
            "freshman" in self.desc or "sophomore" in self.desc):
            self.year = 1
        #Student?
        self.student = 0                            #added 6/3
        if ("student" in self.desc or 
            "studying" in self.desc or
            "estudiante" in self.desc):
            self.student = 1
        if ("students" in self.desc):
            self.student = 0
        #Alum?
        self.alum = 0                               #added 6/3
        if "alum" in self.desc:
            self.alum = 1
        #Occupation?
        self.occupation = 0                         #added 6/5
        for occupation in occupations:
            if occupation in self.desc:
                self.occupation = 1
            if ('aspiring' in self.desc or
                'future' in self.desc):
                self.occupation = 0
        #Emojis
        self.emojis = num_emojis(self.desc)         #added 6/5
        #Parent?
        self.parent = 0                                     #added 6/24
        if("mom" in self.desc or "dad" in self.desc or 
           "mama" in self.desc or "papa" in self.desc or
           "mother" in self.desc or "father" in self.desc):
            self.parent = 1
        #Views My Own?  
        self.views_my_own = 0                                           #added 6/24
        if((( "views" in self.desc or "opinions" in self.desc) and 
            ("my own" in self.desc or "mine" in self.desc)) or 
           ("not endorsement" in self.desc) or ("views my own" in self.desc)
           or ("opinions my own" in self.desc) or ("opinions mine" in self.desc)):
            self.views_my_own = 1
    
    
    #######################################################################
    '''TIME-BASED FEATURES (cont'd)''' 
    #Account Age
    #unit: account age in years
    def account_age(self, file_origin):             #added 6/5, modified 6/10
        seconds = os.path.getctime(file_origin) - self.created_inseconds
        minutes = seconds/60
        hours = minutes/60
        days = hours/24
        years = days/365.25
        return years
    #Tweet Rate
    #unit: tweets per year
    def tweet_rate(self, file_origin):              #added 6/5
        rate = self.tweet_count / self.account_age(file_origin)
        return rate