# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:10:20 2019

"""

import re
import pickle
import argparse
import pandas as pd

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk import word_tokenize

maxlen = 80
max_features = 20000

def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',

                           text)

    text = (re.sub('[\W]+', ' ', text.lower()) +

            ' '.join(emoticons).replace('-', ''))

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-m', '--model', default='weights.010-0.0052_0.0254.hdf5', required=False, help='')    # 必須の引数を追加

    args = parser.parse_args()
    
    print('args:',args)

    X = pd.read_csv("test.csv")
    
    ids = X['id']
    X = X['comment_text']

    X = [preprocessor(strings) for strings in X]
    
    X = [word_tokenize(x) for x in X]
    
    stopWords = stopwords.words('english')
    
    X_nos = []
    for x_elem in X:
        X_nos.append([word.lower() for word in x_elem if word.lower() not in stopWords])
        
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_nos)
    X_final = tokenizer.texts_to_sequences(X_nos)

    model = load_model(args.model)

    x = sequence.pad_sequences(X_final, maxlen=maxlen)

    pred = model.predict(x)

    ids = pd.DataFrame(ids)
    df_pred = pd.DataFrame(pred, columns=['prediction'])
    df_submit = pd.concat([ids, df_pred], axis=1)
    
    df_submit.col = ['id']
    df_submit.to_csv("submission.csv", index=False)
