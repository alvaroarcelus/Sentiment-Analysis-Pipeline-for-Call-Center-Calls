##########################
# TRANSFORMER MODELS
##########################

# Importing libraries 
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import string #remove punctuation
from unidecode import unidecode #remove accents 

##########################
# Installing transformers 
##########################
# ----> TensorFlow
# Transformers will use deep learning libraries such as TensorFlow
# Thus, we must install tensorflow beforehand 

"""
PIPELINE
We are going to use a pretrained pipeline that goes through 3 main processes: 
Tokenizer --> Model fitting --> Postprocessing

LIMITATION 
Most sequence-to-sequence and decoder models have tokenizer limitations 
In our case, the number of tokens should be < 512

SOLUTION 
Thus, we will truncate those conversatoins with more than 512 words to fit 
our constraint by selecting the last 512 words from the conversation. 
"""


# <------- SENTIMENT MODEL

# Importing transformer pipelines pre-trained in Spanish
sentiment = pipeline('sentiment-analysis', model='pysentimiento/robertuito-sentiment-analysis', truncation=True)
classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')

# <------- EMOTION MODEL

def emotion(text_array):
    labels = ['Anger', 'Calmness', 'Happiness', 'Frustration']
    res = classifier(text_array, labels)['labels'][0]
    return res

# <------- INAPROPIATE LANGUAGE MODEL

# Function that given a text, it will detect if there was inapropiate language involved
# if so, it will return those list of words, otherwise "none"
bad_words = list(pd.read_csv('word_vocab/lista_lenguage_inapropiado.txt').PALABROTAS)

def bad_language(text):
    global bad_words
    black_list = []

    # TEXT PREPROCESSING
    #removing stopwords
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text)
    res = [word for word in words if word.lower() not in stop_words]

    # Remove punctuation signs
    res = [word for word in res if word not in string.punctuation]
    # Remove accents 
    spoke = [unidecode(word) for word in res]

    # CHECKING FOR BAD WORDS
    for word in spoke: 
        if word in bad_words: 
            black_list.append(word)
    if len(black_list) == 0:
        black_list = 'None'
    else: 
        #convert array to a string so that we can store it in a df 
        black_list = str(black_list)
    return black_list

# <------- NUISSANCE DETECTION MODEL

# Function that given a text, it will detect if there was any disturbance in from the client's side
# if so, it will return those list of words, otherwise "none"
disturbances = list(pd.read_csv('word_vocab/molestias.txt').ENOJOS)

def nuissance_presence(text):
    global disturbances
    black_list = []

    # TEXT PREPROCESSING
    #removing stopwords
    stop_words = set(stopwords.words('spanish'))
    words = word_tokenize(text)
    res = [word for word in words if word.lower() not in stop_words]

    # Remove punctuation signs
    res = [word for word in res if word not in string.punctuation]
    # Remove accents 
    spoke = [unidecode(word) for word in res]

    # CHECKING FOR BAD WORDS
    for word in spoke: 
        if word in disturbances: 
            black_list.append(word)
    if len(black_list) == 0:
        black_list = 'None'
    else: 
        #convert array to a string so that we can store it in a df 
        black_list = str(black_list)
    return black_list


#################################
# SENTIMENT PIPELINE CHATS 
#################################

# This function uses as input a df that contains the chat jounral of a 
# single, specific conversation. 

def sentiment_pipeline_chat(df):

    #gather all messages per speaker 
    client = list(df[df['client_spk'] == 1]['message'])
    client_nojoin = client.copy()   #copy for emotion model
    client_nojoin = np.vectorize(str)(client_nojoin) # transform all values to str
    client = ''.join(client)

    employee = list(df[df['client_spk'] == 0]['message'])
    employee_nojoin = employee.copy()
    employee_nojoin = np.vectorize(str)(employee_nojoin)
    employee = ''.join(employee)   

    # in the case in which the client/employee did not answer, we add a NaN 
    # and the sentiment analysis clearly would have no result at all 
    # for the client or the employee
    if len(client) == 0: client = 'NaN'  
    if len(employee) == 0: client = 'NaN'  
    
    # ----> SENTIMENT ANALYSIS
    if client != 'NaN':
        res = sentiment(client)
        client_sent = res[0]['label']
        client_scr =  res[0]['score']
    else: 
        client_sent = 'NaN'
        client_scr = 'NaN'

    if employee != 'NaN':
        res = sentiment(employee)
        employee_sent = res[0]['label']
        employee_scr = res[0]['score']
    else:
        employee_sent = 'NaN'
        employee_scr = 'NaN'

    # ----> EMOTION ANALYSIS 
    if len(client_nojoin) > 0:
        client_em = emotion(client_nojoin)
    else:
        client_em = 'NaN'

    if len(employee_nojoin) > 0:
        employee_em = emotion(employee_nojoin)
    else: 
        employee_em = 'NaN'

    # ----> DETECT INNAPROPIATE SPEAKING
    if client != 'NaN':
        client_swear = bad_language(client)
    else:
        client_swear = 'NaN'

    if employee != 'NaN':
        employee_swear = bad_language(employee)
    else: 
        employee_swear = 'NaN'

    # ----> DETECT NUISSANCES
    if client != 'NaN':
        client_disturb = nuissance_presence(client)
    else:
        client_disturb = 'NaN'


    # add new variables to a new dataframe 
    output = pd.DataFrame({'client_sent':[client_sent],
                            'client_scr':[client_scr],
                            'employee_sent':[employee_sent],
                            'employee_scr':[employee_scr],
                            'client_em':[client_em],
                            'employee_em':[employee_em],
                            'client_swear':[client_swear],
                            'employee_swear':[employee_swear],
                            'client_disturb':[client_disturb]
                            })
    return output



#################################
# SENTIMENT PIPELINE CALLS 
#################################

def sentiment_pipeline(data):
    #saving list of emails 
    emails = list(data.email)

    # ----> SENTIMENT ANALYSIS
    clients_sent = sentiment(list(data.client))
    employees_sent = sentiment(list(data.employee))

    client_arr_lbl = []
    employee_arr_lbl = []
    client_arr_scr = []
    employee_scr = []

    for i in range(len(data)):
        #appending all labels 
        client_arr_lbl.append(clients_sent[i]['label'])
        employee_arr_lbl.append(employees_sent[i]['label'])
        #appending all scores
        client_arr_scr.append(clients_sent[i]['score'])
        employee_scr.append(employees_sent[i]['score'])

    # ----> EMOTION ANALYSIS 
    client_em = []
    employee_em = []
    client_swear = []
    employee_swear = []
    client_disturb = []


    for i in tqdm(range(len(data)), desc="Processing", ncols=100):
        client = data.loc[i,'client']
        employee = data.loc[i,'employee']

        if len(client) > 0:
            client_em.append(emotion(client))
        else:
            client_em.append('NaN')

        if len(employee) > 0:
            employee_em.append(emotion(employee))
        else: 
            employee_em.append('NaN')

        # ----> DETECT INNAPROPIATE SPEAKING
        
        if client != 'NaN':
            client_swear.append(bad_language(client))
        else:
            client_swear.append('NaN')

        if employee != 'NaN':
            employee_swear.append(bad_language(employee))
        else: 
            employee_swear.append('NaN')

        # ----> DETECT NUISSANCES
        if client != 'NaN':
            client_disturb.append(nuissance_presence(client))
        else:
            client_disturb.append('NaN')

    #creating dataframe 
    res = pd.DataFrame({'email': emails,
                        'client_label':client_arr_lbl,
                        'client_score': client_arr_scr,
                        'employee_label': employee_arr_lbl,
                        'employee_score': employee_scr,
                        'client_em':client_em,
                        'employee_em':employee_em,
                        'client_swear':client_swear,
                        'employee_swear':employee_swear,
                        'client_disturb':client_disturb}).reset_index(drop=True)

    return res 

















