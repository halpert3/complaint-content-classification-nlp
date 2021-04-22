import pandas as pd
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string

stopwords_list = stopwords.words('english') + list(string.punctuation)
stopwords_list += ["''", '""', '...', '``']
stopwords_list += ['--', 'xxxx']


# date needs to be formatted as 'YYYY-MM-DD' string -- '2021-03-02'
# API allows size max of 1000
def download_and_process_api_data(date, size):
    url = 'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/'

    parameters = {'date_received_min': date,
                  'has_narrative': True,
                  'size': size}

    # get data
    r = requests.get(url, params=parameters)

    # put data in dictionary format
    data = r.json()
    
    # turn data into dataframe
    df = make_dataframe(data)
   
    # loop through each line of 'narrative' in dataframe 
    for i in range(len(df)):
        processed_narr = process_narrative(df['narrative'].loc[i])
        narr = make_lemma_and_concat(processed_narr)
        df['narrative'].loc[i] = narr

    # Consolidate products into 5 categories
    df['product'].replace({'Credit reporting, credit repair services, or other personal consumer reports': 'credit_reporting',
                       'Debt collection': 'debt_collection',
                       'Credit card or prepaid card': 'credit_card',
                       'Mortgage': 'mortgages_and_loans',
                       'Checking or savings account': 'retail_banking',
                       'Money transfer, virtual currency, or money service': 'retail_banking',
                       'Vehicle loan or lease': 'mortgages_and_loans',
                       'Payday loan, title loan, or personal loan': 'mortgages_and_loans',
                       'Student loan': 'mortgages_and_loans'}, inplace=True)

    # replace product names with integers
    product_dict ={'credit_reporting': 0, 'debt_collection': 1, 'mortgages_and_loans': 2, 
               'credit_card': 3, 'retail_banking': 4}
    df['product'].replace(product_dict, inplace=True)

    # Drop useless column that was introduced
    # df.drop([''], axis=1, inplace=True)
    # df.set_index('product')
    return df 


def make_dataframe(data):
    # Instatiate empty dictionary
    new_dict = {}

    # Create empty lists to gather data
    product_list = []
    narrative_list = [] 

    # Loop through data and add to dictionary
    for i in range(len(data['hits']['hits'])):
        product_list.append(data['hits']['hits'][i]['_source']['product'])
        narrative_list.append(data['hits']['hits'][i]['_source']['complaint_what_happened'])
    new_dict['product'] = product_list
    new_dict['narrative'] = narrative_list

    # turn dictionary into dataframe
    return pd.DataFrame.from_dict(new_dict)

# function to tokenize data and remove stopwords
def process_narrative(narrative):
    tokens = nltk.word_tokenize(narrative)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    
    # adding line to remove all tokens with numbers and punctuation
    stopwords_punc_and_numbers_removed = [word for word in stopwords_removed if word.isalpha()]
    
    return stopwords_punc_and_numbers_removed


# function to concat words (used in function below)
def concat_words(list_of_words):
    # remove any NaN's
    # list_of_words = [i for i in list if i is not np.nan]

    concat_words = ''
    for word in list_of_words:
        concat_words += word + ' '
    return concat_words.strip()


# function to lemmatize words and merge each complaint into a single space-separated string



def make_lemma_and_concat(list_of_words):
    lemm = WordNetLemmatizer()

    # remove any NaN's
    list_of_words = [i for i in list_of_words if i is not np.nan]
    
    # lemmatize each word
    lemmatized_list = []
    for idx, word in enumerate(list_of_words):
        lemmatized_list.append(lemm.lemmatize(word))
    
    # make the list into a single string with the words separated by ' '
    concatenated_string = concat_words(lemmatized_list)
    return concatenated_string