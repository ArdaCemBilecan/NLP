import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU,Embedding
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

mark_start = 'ssss '
mark_end = ' eeee'

data_src = []
data_dest = []

for line in open('tur.txt',encoding='UTF-8'):
    en_text , tr_text = line.rstrip().split('\t')
    #adding start and end token
    tr_text = mark_start + tr_text + mark_end
    data_src.append(en_text)
    data_dest.append(tr_text)

print(data_src[200000] , data_dest[200000])
print(len(data_src)) #473035


class TokenizerWrap(Tokenizer):
    def __init__(self,texts,padding,reverse=False , num_words=None):
        Tokenizer.__init__(self,num_words=num_words)
    
        self.fit_on_texts(texts)
        
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
        
        self.tokens = self.texts_to_sequences(texts)
        
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'
            
        self.num_tokens = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.num_tokens) + 2 *np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen = self.max_tokens,
                                           padding = padding,
                                           truncating=truncating)
        
        
    
    
    def token_to_word(self,token):
        word = ' ' if token == 0 else self.index_to_word[token]
        return word
    
    
    def tokens_to_string(self,tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = ' '.join(words)    
        return text
    
    
    def text_to_tokens(self,text,padding,reverse=False):
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)
        
        if reverse:
            tokens = np.flip(tokens , axis = 1)
            truncating = 'pre'
        else:
            truncating = 'post'
            
        
        tokens = pad_sequences(tokens,
                               maxlen=self.max_tokens,
                               truncating = truncating)
        
        return tokens


# for encodder pre-padding
# for decoder post - padding

tokenizer_Src = TokenizerWrap(texts=data_src, padding = 'pre',
                              reverse=True,
                              num_words=None)



tokenizer_Dest = TokenizerWrap(texts=data_dest, padding = 'post',
                              reverse=False,
                              num_words=None)


tokens_src = tokenizer_Src.tokens_padded
tokens_dest = tokenizer_Dest.tokens_padded

print(tokens_src.shape , tokens_dest.shape)
# (473035, 11) (473035, 10)

print(tokens_dest[200000])
print(tokenizer_Dest.tokens_to_string(tokens_dest[200000]))

print(tokens_src[200000])
print(tokenizer_Src.tokens_to_string(tokens_src[200000]))
# this is reverse

token_start = tokenizer_Dest.word_index[mark_start.strip()]
print(token_start) # output:1. 

token_end = tokenizer_Dest.word_index[mark_end.strip()]
print(token_end) # output :2 .
#start token= 1 , end token = 2

        

encoder_input_data = tokens_src

decoder_input_data = tokens_dest[:, :-1]
decoder_output_data = tokens_dest[:, 1:]

print(encoder_input_data[200000])
print(decoder_input_data[200000]) # start token is include
print(decoder_output_data[200000])  #start token is not include

'''
firstly decoder sees '1'. It means 'sssss'
after we expect to generate 'eskik'
other step decoder sees 'eksik' and we want to generate next word 'bir'
It continues like that
'''
  
num_encoder_words = len(tokenizer_Src.word_index) # english total words
num_decoder_words = len(tokenizer_Dest.word_index)# turkish total words
print(num_encoder_words , num_decoder_words)      



#ENCODER
embedding_size = 100

#Glove Vector

word2vec = {}
with open ('glove.6B.100d.txt',encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        

#compare our data

embedding_matrix = np.random.uniform(-1,1,(num_encoder_words,embedding_size))

for word,i in tokenizer_Src.word_index.items():
    if i < num_decoder_words:
        embedding_vector = word2vec.get(word)
        if embedding_matrix is not None:
            embedding_matrix[i] = embedding_vector


# Encoder Model
encoder_input = tf.keras.Input(shape=(None,), name='encoder_input')
encoder_embedding = Embedding(input_dim=num_encoder_words,
                              output_dim = embedding_size,
                              weights=[embedding_matrix],
                              trainable = True,
                              name='encoder_embedding')

state_size = 256

encoder_gru1 = GRU(state_size , name='encoder_gru1',return_sequences=True)
encoder_gru2 = GRU(state_size , name='encoder_gru2',return_sequences=True)
encoder_gru3 = GRU(state_size , name='encoder_gru3',return_sequences=False) 

def connect_encoder():
    net = encoder_input
    net = encoder_embedding(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
    encoder_output = net
    return encoder_output


encoder_output = connect_encoder()
        
    
# DECODER

decoder_initial_state = tf.keras.Input(shape=(state_size,),
                                         name = 'decoder_initial_sstate')

decoder_input = tf.keras.Input(shape=(None,),
                               name = 'decoder_input')


decoder_embedding = Embedding(input_dim=num_decoder_words,
                              output_dim = embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size,name='decoder_gru1',return_sequences=True)   
decoder_gru2 = GRU(state_size,name='decoder_gru2',return_sequences=True)    
decoder_gru3 = GRU(state_size,name='decoder_gru3',return_sequences=True)

decoder_dense = Dense(num_decoder_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(initial_state):
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net,initial_state = initial_state)    
    net = decoder_gru2(net,initial_state = initial_state)  
    net = decoder_gru3(net,initial_state = initial_state)
    
    decoder_output = decoder_dense(net)
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = tf.keras.Model(inputs=[encoder_input , decoder_input],
                    outputs=[decoder_output])
    
    
    
model_encoder = tf.keras.Model (inputs=[encoder_input],
                                outputs=[encoder_output])   

decoder_output = connect_decoder(initial_state = decoder_initial_state)
    
model_decoder = tf.keras.Model(inputs=[decoder_input,decoder_initial_state],
                               outputs=[decoder_output])
    
    

def sparse_cross_entropy(y_true,y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    loss_mean = tf.reduce_mean(loss)
    
    return loss_mean


optimizer = RMSprop(lr=1e-3)


model_train.compile(optimizer=optimizer , 
                    loss=sparse_cross_entropy)      
    
    
path='checckpoint.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(path,save_weights_only=True)



x_data = {'encoder_input':encoder_input_data , 
          'decoder_input': decoder_input_data}

y_data = {'decoder_output':decoder_output_data}

model_train.fit(x=x_data , y=y_data , batch_size =256, epochs=10, 
                callbacks=[checkpoint])

    
    


def translate(input_text, true_output_text=None):
    input_tokens = tokenizer_Src.text_to_tokens(text=input_text,
                                               reverse=True,
                                                    padding='pre')
    initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_Dest.max_tokens
    decoder_input_data = np.zeros(shape=(1, max_tokens), dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'decoder_initial_state': initial_state, 'decoder_input': decoder_input_data}
        decoder_output = model_decoder.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        sampled_word = tokenizer_Dest.token_to_word(token_int)
        output_text += ' ' + sampled_word
        count_tokens += 1
        print('Input text:')
        print(input_text)
        print('Translated text:')
        print(output_text)
        if true_output_text is not None:
            print('True output text:')
            print(true_output_text)
         
    
    
    
    
    
    
    
    
    
            
            
        
        
        
