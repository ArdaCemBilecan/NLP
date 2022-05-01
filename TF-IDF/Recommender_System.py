import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(g['name'].split()) for g in genres )  # Seeing all genres names for a movie in a string
    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(k['name'].split()) for k in keywords ) # seeing all keyword names for a movie in a string

    return "%s  %s" % (genres,keywords)
    # return example : 'Action Adventure Fantasy ScienceFiction  cultureclash future spacewar spacecolony society spacetravel 
        #futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d'

    

df = pd.read_csv('tmdb_5000_movies.csv')
df['string'] = df.apply(genres_and_keywords_to_string,axis=1)

tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['string'])
print(X)

movie2idx = pd.Series(df.index , index = df['title'])
print(movie2idx)
idx = movie2idx['Scream 3']
print(idx)

query = X[idx]
print(query.toarray())

scores = cosine_similarity(query,X) 
print(scores)

scores = scores.flatten() # 1xN array  means 1-D array
print(scores)

#plt.plot(scores) just wanna see the scores distribution
print((-scores).argsort()) # we want to descending order. Because in cosine_similarity; 
#the less the angle between the 2 vectors, the more similar they are.

recommended_idx = (-scores).argsort()[1:6] # wanna see just top 5.  0. index is same movie.
print(df['title'].iloc[recommended_idx])
# Briefly we created a function what we did upside
def recommend(title):
    idx = movie2idx[title]
    if type(idx) == pd.Series:
        idx = idx.iloc[0]
    
    query = X[idx]
    scores = cosine_similarity(query,X)
    scores = scores.flatten()
    recommended_idx = (-scores).argsort()[1:6]
    return df['title'][recommended_idx]

print('Recommendations for Scream3 : ')
print(recommend('Scream 3'))  

print('Recommendations for Mortal Kombat : ')
print(recommend('Mortal Kombat'))  

print("Recommendations for  Pirates of the Caribbean: At World's End : ")
print(recommend("Pirates of the Caribbean: At World's End"))  







