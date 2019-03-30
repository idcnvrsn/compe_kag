# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:44:35 2019

"""

'''
#Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

**Notes**

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

'''
#from __future__ import print_function
import pickle
import re
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk import word_tokenize
#import nltk
#nltk.download()
import pandas as pd

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer

do_cache = True

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',

                           text)

    text = (re.sub('[\W]+', ' ', text.lower()) +

            ' '.join(emoticons).replace('-', ''))

    return text


print('Loading data...')

if do_cache:
    train_text = pd.read_csv(r"train.csv")
    
    train_text = train_text.iloc[:10]
    
    y = train_text["target"]

    X = train_text["comment_text"]

    X = [preprocessor(strings) for strings in X]
    
    X = [word_tokenize(x) for x in X]
    
    stopWords = stopwords.words('english')
    
    X_nos = []
    for x_elem in X:
        X_nos.append([word.lower() for word in x_elem if word.lower() not in stopWords])
        
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_nos)
    X_final = tokenizer.texts_to_sequences(X_nos)

    with open('X_final.pkl', 'wb') as f:
        pickle.dump(X_final, f)
    with open('y_final.pkl', 'wb') as f:
        pickle.dump(y, f)

    import sys
    sys.exit()
else:
    with open('X_final.pkl', 'rb') as f:
        X_final = pickle.load(f)    

    with open('y.pkl', 'rb') as f:
        y = pickle.load(f)    

x_train, x_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

"""
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x_train)
print(vectorizer.get_feature_names())
print(X.toarray())
"""

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=40,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
