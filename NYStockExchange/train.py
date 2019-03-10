# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:05:11 2018

"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

df = pd.read_csv("prices-split-adjusted.csv")

df = df.head(int(df.shape[0]/1000))

df_close = df[["close"]]

df_close=df_close / df_close.max()

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)

length_of_sequences = 100
(X_train, y_train), (X_test, y_test) = train_test_split(df_close, test_size=0.2, n_prev =length_of_sequences)

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

in_out_neurons = 1
hidden_neurons = 600

model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
model.add(Dense(in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="Adam")#"rmsprop")

dir_name = datetime.now().strftime('%Y%m%d_%H%M%S')
os.mkdir(dir_name)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
filepath = dir_name + os.sep + 'weights.{epoch:03d}-{loss:.4f}_{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

model.fit(X_train, y_train, batch_size=300, epochs=35, callbacks=[es, mcp], validation_split=0.05) 

predicted = model.predict(X_test) 

dataf =  pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
dataf.plot(figsize=(15, 5))


