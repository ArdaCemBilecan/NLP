import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU,Embedding
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('hepsiburada.csv')
print(dataset.head())

target = dataset['Rating'].values.tolist()
comments = dataset['Review'].values.tolist()

print(len(comments) , len(target))

x_train , x_test , y_train , y_test = train_test_split(comments , target , test_size = 0.33)

print(len(x_train) , len(x_test))

# Tokenizer
num_words = 50000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(comments)
print(tokenizer.word_index)


# text to index
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

# Calculate number of tokens

num_tokens = [len(tokens) for tokens in x_train_tokens+x_test_tokens]
num_tokens = np.array(num_tokens)

print(np.mean(num_tokens)) # Each comments include approx 23 words.
print(np.max(num_tokens)) # Found max number of token , The longest comment

index = np.argmax(num_tokens)

# Calculating , Matrix elements number
max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)

print(np.sum(num_tokens < max_tokens) / len(num_tokens))
# 96% comments are not longer than 66


# Padding
x_train_pad = pad_sequences(x_train_tokens,maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens,maxlen=max_tokens)

print(x_train_pad.shape , x_test_pad.shape)


# index to string
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values() , idx.keys()))

def token_to_string(tokens):
    words = [ inverse_map[token] for token in tokens if token !=0]
    text = ' '.join(words)
    return text

# try to function
print(x_train[800])
print("--------")
print(token_to_string(x_train_tokens[800]))

x_train_pad = np.array(x_train_pad)
x_test_pad = np.array(x_test_pad)
y_train = np.array(y_train)
y_test = np.array(y_train)


# Creating GRU Model

embedding_size=50
model = Sequential()
model.add(Embedding(input_dim=num_words , 
                    output_dim=embedding_size,
                   input_length = max_tokens,
                   name = 'embedding_layer'))

model.add(GRU(units=16,return_sequences=True))
model.add(GRU(units=8,return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1 , activation='sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

model.summary()

model.fit(x_train_pad , y_train , epochs=10 , batch_size = 256) 

result = model.evaluate(x_test_pad , y_test) 
print(result[1]) # Accuracy:  0.9025 
