import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
#nltk.download('punkt')
word2idx = {}
doc_as_int = []
tokenized_docs = []

def Index_Mapping(df):
    idx = 0
    for doc in df['text']:
        words = word_tokenize(doc.lower())
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx+=1
            doc_as_int.append(word2idx[word])
        tokenized_docs.append(doc_as_int)


df = pd.read_csv('bbc_text_cls.csv')

Index_Mapping(df)

#reverse mapping
idx2word = {v:k for k , v in word2idx.items()}

N = len(df['text']) #number of documents
V = len(word2idx) # number of words

# Calculating TF-IDF:  TF means that how many times the word appears  in the document
# IDF means that How many documents include this word in
tf = np.zeros((N,V))
#populate term-frequency counts

for i , doc_as_int in enumerate(tokenized_docs):
    for j in doc_as_int:
        tf[i,j] +=1

#compute idf
document_freq = np.sum(tf>0, axis =0) # document frequency 
idf = np.log(N/document_freq)

# compute TF-IDF
tf_idf = tf*idf

i = np.random.choice(N)
row = df.iloc[i]
print('Label: ',row['labels'])
print('Text: ',row['text'].split("\n",1)[0])
print('Top 5 Terms:')
scores = tf_idf[i]
indicies = (-scores).argsort()
for j in indicies:
    print(idx2word[j])
