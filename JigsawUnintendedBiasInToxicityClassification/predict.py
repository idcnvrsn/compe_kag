# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:25:28 2019

"""
import pickle
import argparse
from keras.models import load_model
from keras.preprocessing import sequence

maxlen = 80

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-m', '--model', default='weights.010-0.0052_0.0254.hdf5', required=False, help='')    # 必須の引数を追加

    args = parser.parse_args()
    
    print('args:',args)

    with open('X_final.pkl', 'rb') as f:
        X_final = pickle.load(f)    

    model = load_model(args.model)

    x = sequence.pad_sequences(X_final, maxlen=maxlen)

    pred = model.predict(x)
