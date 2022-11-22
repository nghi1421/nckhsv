import io
import json


import numpy as np
import pandas as pd

import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors

Data_Full_train_2C = "MyData/c4_label3.xlsx"
url_word2vec_aspect = "word2vec/dataset_HDH_w2v.model"

EMBEDDING_DIM = 300
NUM_WORDS = 50000
max_length = 300
pad = ['post', 'pre']
test_num_full = 400

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_temp():
    # test_data = pd.read_excel(Data_Full_train_2C, 'Sheet1')
    train_data = pd.read_excel(Data_Full_train_2C, 'c4_pcbn_ch')
    train_len = len(train_data)

    print(train_data.isnull().sum())
    print(train_len)

    # dic = {'CH': 0, 'CHT': 1, 'GT': 2, 'KN': 3, 'MT': 4}
    # dic = {'BNA': 0, 'CPKGF': 1, 'CTHDH': 2, 'DB_TT': 3, 'DC_VDLQ': 4,'DP_TT': 5,'KN_TT': 6,'KNHDH': 7,'KTMR': 8,'PCBN': 9,'PDBN': 10,'PTBN': 11,'QLKG': 12,'TQ': 13,'TT': 14,'TT_TN': 15}
    # dic = {'c1': 0, 'c2': 1, 'c3': 2, 'c4': 3}
    # dic = {'knhdh':0, 'cthdh':1}
    # dic = {'cht':0, 'none':1}
    dic = {'none':0, 'ch':1};
    # dic = {'kn':0,'mt':1}
    # dic = {'kn':0,'mt':1}
    labels = train_data.label3.apply(lambda x: dic[x])



    val_data = train_data.sample(frac=0.2, random_state=42)  # cross validation: 1/10
    test_data = train_data.sample(frac=0.2, random_state=42)    
    # test_data = train_data[515:714]
    # train_data = train_data.drop(test_data.index)
    # val_data = train_data[315:514]
    # train_data = train_data.drop(val_data.index)
    texts = train_data.text
    '''
    print("texts")
    print(len(texts))
    print("len train_data")
    print(len(train_data))
    print("train_data 7000:")
    print(train_data.text[7000])
    print("val_data 0:")
    print(val_data.text[val_data_raw_from])
    '''
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                          lower=True)
    tokenizer.fit_on_texts(texts)
    tokenizer_json = tokenizer.to_json()
    with io.open('word2vec/vocab.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    # tokenizer_json = tokenizer.to_json()
    # with io.open('D:\\Jobs\\HV BCVT\\Do an TN\\HK1 - 2020-2021\\Le Thi Hong Anh\\Resources\\vocab.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    print("tokenizer Text:")
    print(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data.text)
    sequences_test = tokenizer.texts_to_sequences(test_data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    '''
    print("Cau thu train_data")
    print(sequences_train[train_len - test_num_full + 1])
    '''
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding=pad[0])
    X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1], padding=pad[0])
    X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1], padding=pad[0])
    y_train = to_categorical(np.asarray(labels[train_data.index]))
    y_val = to_categorical(np.asarray(labels[val_data.index]))
    y_test = to_categorical(np.asarray(labels[test_data.index]))

    word_vectors = KeyedVectors.load(url_word2vec_aspect, mmap='r')

    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    print("vocabulary_size size:")
    print(vocabulary_size)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    print("embedding_matrix size:")
    print(embedding_matrix)
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    del (word_vectors)

    from keras.layers import Embedding
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)
    print("embedding_layer")
#    print(embedding_layer.weights())
    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    '''
    X_test = X_train[train_len - test_num_full:]
    y_test = y_train[train_len - test_num_full:]
    X_train = X_train[0: train_len - test_num_full]
    y_train = y_train[0: train_len - test_num_full]
    '''
    return X_train, y_train, X_test, y_test, X_val, y_val, embedding_layer


