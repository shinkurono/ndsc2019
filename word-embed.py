import numpy as np
import pandas as pd
import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()

def preprocessing(titles_array):
    processed_array = []
    for title in tqdm(titles_array):
        processed = re.sub('[^a-zA-Z ]', '', title)
        words = processed.split()
        processed_array.append(' '.join([word for word in words if len(word) > 1]))

    return processed_array

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train['processed'] = preprocessing(df_train['title'])
df_test['processed'] = preprocessing(df_test['title'])

sentences = pd.concat([df_train['processed'], df_test['processed']],axis=0)
train_sentences = list(sentences.progress_apply(str.split).values)

model = Word2Vec(sentences=train_sentences, 
                 sg=1, 
                 size=400,  
                 workers=4)

model.wv.save_word2vec_format('custom_glove_100d.txt')