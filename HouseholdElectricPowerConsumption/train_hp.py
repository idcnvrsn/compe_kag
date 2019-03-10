# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:05:11 2018

https://www.kaggle.com/uciml/electric-power-consumption-data-set
を解くスクリプト
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler
 
from hyperopt import hp, tpe, Trials, fmin, space_eval

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import History

hyperopt_parameters = {
    'length_of_sequences' : hp.quniform('length_of_sequences',50, 150, 10),
    'batch_size': hp.quniform('batch_size', 2, 32, 2),
    'hidden_neurons': hp.quniform('hidden_neurons', 100, 800,100),
#    'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
}


def _load_data(data, n_prev = 30):  
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

def objective(args):
    print(args)
    batch_size = int(args['batch_size'])
    length_of_sequences = int(args['length_of_sequences'])
    hidden_neurons = int(args['hidden_neurons'])

    (X_train, y_train), (X_test, y_test) = train_test_split(df_close, test_size=0.2, n_prev =length_of_sequences)

    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
    model.add(Dense(in_out_neurons))  
    model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="Adam")#"rmsprop")
        
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    filepath = dir_name + os.sep + str(trial_num) + '_weights.{epoch:03d}-{loss:.4f}_{val_loss:.4f}.hdf5'
    mcp = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=30, callbacks=[es, mcp], validation_split=0.05) 
#    import pdb;pdb.set_trace()
    return history.history['val_loss'][0]


if __name__ == '__main__':」の意味 - Qiita

    csv_filename = r"C:\Users\kodama\Documents\github\compe_kag\HouseholdElectricPowerConsumption\household_power_consumption.txt"
    df = pd.read_csv(csv_filename, sep = ";")
    
    df = df.iloc[:,2:]
    df = df.head(int(df.shape[0]/1000))
    
    ms = MinMaxScaler()
    data_norm = ms.fit_transform(df)
    
    df_close = df["Global_active_power"]
    
    df_close=df_close / df_close.max()
    
    
    dir_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(dir_name)
    
    
    
    #length_of_sequences = 100
    
    
    in_out_neurons = 1
    #hidden_neurons = 600


    # iterationする回数
    max_evals = 100
    # 試行の過程を記録するインスタンス
    trials = Trials()
    
    trial_num = 1
    while trial_num < max_evals:
        best = fmin(
            # 最小化する値を定義した関数
            objective,
            # 探索するパラメータのdictもしくはlist
            hyperopt_parameters,
            # どのロジックを利用するか、基本的にはtpe.suggestでok
            algo=tpe.suggest,
            max_evals=trial_num,
            trials=trials,
            # 試行の過程を出力
            verbose=1
        )
        
        with open(dir_name + os.sep + "trials.pkl","wb") as f:    
            pickle.dump(trials, f)
        
        trial_num += 1
        
    
    predicted = model.predict(X_test) 
    
    dataf =  pd.DataFrame(predicted[:200])
    dataf.columns = ["predict"]
    dataf["input"] = y_test[:200]
    dataf.plot(figsize=(15, 5))
    
    
