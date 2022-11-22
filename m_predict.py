import json
import re
import numpy as np
import pandas as pd
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.models import Model
from keras import regularizers
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras_preprocessing.text import tokenizer_from_json
from keras import regularizers

url_full_train_data = "D:\\Resources\\Corpus_full_train.xls"
url_word2vec_full = "D:\\Resources\\HealthPlus_Full_Data_for_Word2vec_14122020.model"
val_data_full_from = 29000  # 3001
val_data_full_to = 29001  # 6003
pad = 'post'
drop = 0.2
epoch = 20
batch_size = 128
max_length = 300
NUM_WORDS = 50000
EMBEDDING_DIM = 300
test_num_full = 3004
num_filters = 300
activation_func = "relu"
L2 = 0.004
filter_sizes = [3, 4, 5]

str = "xe phiên_bản cũ cảm_giác tiện_nghi kém, nhìn quá_chán"
similar = []

aspect_data = pd.read_excel(url_sentiment_data, 'Sheet1', encoding='utf-8')
for i in range(0, len(aspect_data)):
    similar.append(aspect_data.senti_item[i])

str = str.replace('_',' ')
str = str.replace(',','')


def clean_str(string):
    strtemp = string
    strtemp = strtemp.replace('(', '')
    strtemp = strtemp.replace(')', '')
    strtemp = strtemp.replace(',', ', ')
    strtemp = strtemp.replace('  ', ' ')
    strtemp = strtemp.replace('!', '.')
    strtemp = strtemp.replace('?', '.')
    strtemp = strtemp.replace(';', '.')
    strtemp = strtemp.replace('\n', '.')
    strtemp = strtemp.replace(' .', '.')
    strtemp = strtemp.replace('. ', '.')
    strtemp = strtemp.replace('.', ' . ')
    return strtemp.strip().lower()


def standardize_doc(text):
    strtext = text
    strtext = strtext.replace('.', ' ')
    strtext = " ".join(strtext.split())
    '''
    str = str.replace('    ', ' ')
    str = str.replace('   ', ' ')
    str = str.replace('  ', ' ')
    '''
    return strtext


def sentence_split(text):
    strtemp = text
    strtemp = strtemp.replace('(', '')
    strtemp = strtemp.replace(')', '')
    strtemp = strtemp.replace(',', ', ')
    strtemp = strtemp.replace('  ', ' ')
    strtemp = strtemp.replace('!','.')
    strtemp = strtemp.replace('?', '.')
    strtemp = strtemp.replace(';', '.')
    strtemp = strtemp.replace('\n', '.')
    strtemp = strtemp.replace(' .', '.')
    strtemp = strtemp.replace('. ', '.')
    arr = strtemp.split('.')
    return arr



model_doc.load_weights('D:\\Resources\\weight\\CNN_doc_raw_train_2c-021-0.0426-0.9520_0.9619.h5')
# okay here's the interactive part
text = "nội thất (nhất là bộ ghế) quá chán! ngoại thất thì cũng ngon! động cơ xịn không?"
doc_token = []
sentences = []
while True:
    evalSentence = input()
    if evalSentence:
        # evalSentence = evalSentence.lower()
        evalSentence = clean_str(evalSentence)
        evalSentence = standardize_doc(evalSentence)
        print(evalSentence)
    else:
        break
    testArr = convert_text_to_index_array(evalSentence)

    doc_token.append(testArr)
    doc_token = pad_sequences(doc_token, maxlen=300, padding='post')
    pred = model_doc.predict(doc_token)
    # print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    print("polarity: %s" % (aspect_labels[np.argmax(pred)]))

    del evalSentence
    doc_token = []
