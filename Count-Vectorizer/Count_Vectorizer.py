import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import wordnet

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')

df = pd.read_csv('bbc_text_cls.csv')

inputs = df['text']
labels = df['labels']

labels.hist(figsize=(10,5))
# Scenario 1 , no any adding hyper parameter
x_train , x_test , y_train , y_test = train_test_split(inputs,labels,test_size=0.25,random_state=33)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(X_train, y_train)
print('Train Score is: ',model.score(X_train,y_train)) #0.9928
print('Test Score is: ',model.score(X_test,y_test)) # 0.9748

# Scenario 2 , with stopwords
vectorizer = CountVectorizer(stop_words = 'english')
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)
model = MultinomialNB()
model.fit(X_train, y_train)
print('Train Score is: ',model.score(X_train,y_train)) # 0.9940
print('Test Score is: ',model.score(X_test,y_test)) # 0.9748



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Scenario 3 , with Lemmatization     
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self,doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        
        return [self.wnl.lemmatize(word,pos=get_wordnet_pos(tag)) for word , tag in words_and_tags]
    
    
vectorizer = CountVectorizer(tokenizer =LemmaTokenizer())
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test) 
model = MultinomialNB()
model.fit(X_train, y_train)
print('Train Score is: ',model.score(X_train,y_train)) # 0.9934
print('Test Score is: ',model.score(X_test,y_test)) # 0.9746



# Scenario 3 , with Stemming 
class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
    
    def __call__(self,doc):
        tokens = word_tokenize(doc)
        return [self.porter.stem(token) for token in tokens]

    
vectorizer = CountVectorizer(tokenizer =StemTokenizer())
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test) 
model = MultinomialNB()
model.fit(X_train, y_train)
print('Train Score is: ',model.score(X_train,y_train)) # 0.9916
print('Test Score is: ',model.score(X_test,y_test)) # 0.9748    
    
    

# Scenario 4 , with split 
def simple_tokenizer(s):
    return s.split()
    
vectorizer = CountVectorizer(tokenizer =simple_tokenizer)
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test) 
model = MultinomialNB()
model.fit(X_train, y_train)
print('Train Score is: ',model.score(X_train,y_train)) # 0.9964
print('Test Score is: ',model.score(X_test,y_test)) # 0.9640    
    
    
    
    
    
    
    
    
    

