# import libraries
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level='DEBUG')

# import dataset
df = pd.read_csv(r'email_spam_classifier\Sources\spam.csv', encoding='latin-1')
logging.debug(df.info())

# dropping unnessary column
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True) # new


# renaming the columns
df.rename(columns={'v1':'target', 'v2':'sms'}, inplace=True) # new


# label encoding the target column to get 0 or 1 for ham and spam
from sklearn.preprocessing import LabelEncoder # new
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
logging.debug(df.head(5))

# checking for null values
logging.debug(df.isnull().sum()) # new

# check for duplicates and droping them
df = df.drop_duplicates(keep='first')
logging.debug(df.duplicated().sum())

# EDA : Exploratory data analysis
# import matplotlib.pyplot as plt
# plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')
# plt.show()

# counting number of character, words, sentence in a record
import nltk
# nltk.download('punkt') # new

# count chars, words and sentences
df['num_chars'] = df.sms.apply(len)
df['num_words'] = df.sms.apply(lambda x: len(nltk.word_tokenize(x))) # new
df['num_sents'] = df.sms.apply(lambda x: len(nltk.sent_tokenize(x))) # new
logging.debug(df)
logging.debug(f"describing the newly added cols\n {df[['num_chars', 'num_words', 'num_sents']].describe()}")
logging.debug(f"describing the newly added cols for hams\n {df[df.target==0][['num_chars', 'num_words', 'num_sents']].describe()}") # new
logging.debug(f"describing the newly added cols for spams\n {df[df.target==1][['num_chars', 'num_words', 'num_sents']].describe()}") # new


# analysing the above part with graphs
import seaborn as sns
sns.histplot(df[df.target == 0]['num_chars'])

