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

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

with open((r"training_text"),"r",encoding="utf-8") as f:
    train_text = f.readlines()
train_text = train_text[1:]

train_text = [text.split("||")[1] for text in train_text]
train_text = [text[0:-1] for text in train_text]

with open((r"training_variants"),"r",encoding="utf-8") as f:
    train_variants = f.readlines()
train_variants = train_variants[1:]

train_variants = [int(variant.split(",")[-1]) for variant in train_variants]

X = train_text
y = to_categorical(train_variants)

X=X[:1]
y=y[:1]

from nltk.corpus import stopwords
from nltk import word_tokenize

#import nltk
#nltk.download()

X = [word_tokenize(x) for x in X] 

stopWords = stopwords.words('english')

[word.lower() for word in X[0] if word.lower() in stopWords]


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)




x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
model.add(Embedding(max_features, 32))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
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
