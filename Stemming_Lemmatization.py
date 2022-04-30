import nltk
from nltk.stem import PorterStemmer

porter = PorterStemmer()
print(porter.stem('walking')) # walk
print(porter.stem('walks')) # walk
print(porter.stem('ran')) # ran
print(porter.stem('running')) # run
print(porter.stem('replacement')) #replac
print(porter.stem('unnecessary')) # unnecessari

sentence = 'Lemmatization is more sophisticated than stemming'.split()
for word in sentence:
    print(porter.stem(word), end=' ') # lemmat is more sophist than stem 
    
    
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('walking')) # walking
print(lemmatizer.lemmatize('walking' , pos = wordnet.VERB)) #walk
print(lemmatizer.lemmatize('walks'))  # walk
print(lemmatizer.lemmatize('going'))  #going
print(lemmatizer.lemmatize('going' , pos = wordnet.VERB)) #go
print(lemmatizer.lemmatize('ran')) #ran
print(lemmatizer.lemmatize('ran' , pos = wordnet.VERB)) # run


print(lemmatizer.lemmatize('was' , pos = wordnet.VERB)) # be
print(porter.stem('was')) # wa

print(lemmatizer.lemmatize('is' , pos = wordnet.VERB)) # be
print(porter.stem('is')) # is

print(lemmatizer.lemmatize('better' , pos = wordnet.ADJ)) # good
print(porter.stem('better')) # better


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
    
#nltk.download('averaged_perceptron_tagger')
sentence2 = 'Donald Trump has a devoted following'.split()

words_and_tags = nltk.pos_tag(sentence2)
print(words_and_tags)

for word,tag in words_and_tags:
    lemma = lemmatizer.lemmatize(word , pos =get_wordnet_pos(tag))
    print(lemma , end=' ')    # Donald Trump have a devote FOLLOWING


sentence3 = 'The cat was following the bird  as it flew by'.split()
words_and_tags2 = nltk.pos_tag(sentence3)
print(words_and_tags2)

for word,tag in words_and_tags2:
    lemma = lemmatizer.lemmatize(word , pos =get_wordnet_pos(tag))
    print(lemma , end=' ')    #The cat be FOLLOW the bird a it fly by 







