###################################
# AUDIO CLEANSING FUNCTIONS 
###################################

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
# wordcloud
from wordcloud import WordCloud
#nlp library 
import nltk 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from unidecode import unidecode
#regular expressions 
import re


def remove_accents(word):
    # Transliterate Unicode characters to their closest ASCII equivalents
    word = unidecode(word)
    # Remove non-alphabet characters using regular expression
    word = re.sub(r'[^A-Za-z]', '', word)
    return word

def remove_stopwords(text):
    # we add more common stopwords to the standard list 
    stop_words = list(set(stopwords.words('spanish')))
    stop_words.extend(['si','sí','no','por','ok','bueno','dia','día','pues','entonces','ajá','ha','ah','tal','va','sé','hay','di','da','ve'])
    global stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# function that given a string, it will clean it and return a dataframe with everyt word's frequency

def process_text(text):
    freq = pd.DataFrame({'word':[], 'frequency':[]})
    # removing stopwords 
    text = remove_stopwords(text)

    # tokenizing text and changing all words to lowe case
    tokens = word_tokenize(text.lower())

    # Define regular expression pattern to match unwanted characters
    pattern = re.compile(r'[^\w\s]')  # Matches any character that is not a word character or whitespace

    # Remove unwanted characters and create token list
    tokens = [re.sub(pattern, '', token) for token in tokens if not re.match(pattern, token)]

    # Remove accents from words
    new_token=[]
    for word in tokens: 
        new_token.append(remove_accents(word))
    tokens = new_token

    # Calculating frequency of each word --> the output is a dictionary 
    freq_dist = FreqDist(tokens)

    #transforming nltk dictionary to pandas dataframe 
    data = pd.DataFrame.from_dict(dict(freq_dist), orient='index', columns=['frequency']).reset_index()
    data = data.rename(columns={'index':'word'})
    freq = pd.concat([freq, data])
    return freq


def global_freq(person,df):
    freq_df = pd.DataFrame({'word':[], 'frequency':[]})
    for i in range(len(df)):
        txt = df.loc[i,person]
        res = process_text(txt)
        freq_df = pd.concat([freq_df,res])
    
    #grouping by each word 
    freq_df = freq_df.groupby(['word']).sum().reset_index().sort_values('frequency', ascending=False).reset_index(drop=True)
    return freq_df