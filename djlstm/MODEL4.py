
# coding: utf-8

# # Predicting the Dow Jones with News
# The goal of this project is to use the top, daily news headlines from Reddit to predict the movement of the Dow Jones Industrial Average. The data for this project spans from 2008-08-08 to 2016-07-01, and is from this [Kaggle dataset](https://www.kaggle.com/aaron7sun/stocknews). 
# The news from a given day will be used to predict the difference in opening price between that day, and the following day. This method is chosen because it should allow all of the day's news to be incorporated into the price compared to closing price, which could only incorporate the day's morning and afternoon news.
# For this project, we will use GloVe to create our word embeddings and CNNs followed by LSTMs to build our model. This model is based off the work done in this paper https://www.aclweb.org/anthology/C/C16/C16-1229.pdf.

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import initializers
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input
from keras.layers import Dense, merge, BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers import *
from keras.layers import add
import time
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
nltk.download('stopwords')
from PredictGenerator import model_show_predictions
import plotter

"""

dj = pd.read_csv("../data/DowJones.csv") #read in stock prices
news = pd.read_csv("../data/News.csv") #read in news data
dj.head()
dj.isnull().sum()
news.isnull().sum()
news.head()

print("Dow Jones Raw Data Shape: " + str(dj.shape))
print("Raw News Data Shae: " + str(news.shape))

# Compare the number of unique dates. We want matching values.
print("Number of Unique Dates in DJ data set: " + str(len(set(dj.Date))))
print("Number of Unique Dates in news headlines: " + str(len(set(news.Date))))

# Remove the extra dates that are in news
news = news[news.Date.isin(dj.Date)]

print("Removed extra dates in DJ data set: " + str(len(set(dj.Date))))
print("Removed extra dates in DJ data set: " + str(len(set(news.Date))))

# Calculate the difference in opening prices between the following and current day.
# The model will try to predict how much the Open value will change beased on the news.
dj = dj.set_index('Date').diff(periods=1)
dj['Date'] = dj.index
dj = dj.reset_index(drop=True)

dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1) # Remove unneeded features from the dataset

dj.head()
dj = dj[dj.Open.notnull()] # Remove top row since it has a null value.
dj.isnull().sum() # Check if there are any more null values.

# Create a list of the opening prices and their corresponding daily headlines from the news
price = []
headlines = []

for row in dj.iterrows():
    daily_headlines = []
    date = row[1]['Date']
    price.append(row[1]['Open'])
    for row_ in news[news.Date==date].iterrows():
        daily_headlines.append(row_[1]['News'])
    
    # Track progress
    headlines.append(daily_headlines)
    if len(price) % 500 == 0:
        print(len(price))


# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# FUNCTION clean_text = The purpose of this funciton is to remove unwanted characters
# and format the text to create fewer null words embeddings
def clean_text(text, remove_stopwords = True):
    
    text = text.lower() # Convert words to lower case
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

# Clean the headlines
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)


# Take a look at some headlines to ensure everything was cleaned well
print("VERIFY CLEANING TOOK PLACE: " + str(clean_headlines[0]))

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

for date in clean_headlines:
    for headline in date:
        for word in headline.split():
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
            
print("Size of Vocabulary:", len(word_counts))


# LOAD GLOVE EMBEDDINGS
embeddings_index = {}
with open('../data/glove.840B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding


print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from GloVe, and are used more than our threshold.
missing_words = 0
threshold = 10

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from GloVe:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe
#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total Number of Unique Words:", len(word_counts))
print("Number of Words we will use:", len(vocab_to_int))
print("Percent of Words we will use: {}%".format(usage_ratio))

# Need to use 300 for embedding dimensions to match GloVe's vectors.
embedding_dim = 300

nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim))
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in GloVe, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


# Note: The embeddings will be updated as the model trains, so our new 'random' embeddings will be more accurate by the end of training. This is also why we want to only use words that appear at least 10 times. By having the model see the word numerous timesm it will be better able to understand what it means. 

# Change the text from words to integers
# If word is not in vocab, replace it with <UNK> (unknown)
word_count = 0
unk_count = 0

int_headlines = []

for date in clean_headlines:
    int_daily_headlines = []
    for headline in date:
        int_headline = []
        for word in headline.split():
            word_count += 1
            if word in vocab_to_int:
                int_headline.append(vocab_to_int[word])
            else:
                int_headline.append(vocab_to_int["<UNK>"])
                unk_count += 1
        int_daily_headlines.append(int_headline)
    int_headlines.append(int_daily_headlines)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

# Find the length of headlines
lengths = []
for date in int_headlines:
    for headline in date:
        lengths.append(len(headline))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

lengths.describe()

# Limit the length of a day's news to 200 words, and the length of any headline to 16 words.
# These values are chosen to not have an excessively long training time and 
# balance the number of headlines used and the number of words from each headline.
max_headline_length = 16
max_daily_length = 200
pad_headlines = []

for date in int_headlines:
    pad_daily_headlines = []
    for headline in date:
        # Add headline if it is less than max length
        if len(headline) <= max_headline_length:
            for word in headline:
                pad_daily_headlines.append(word)
        # Limit headline if it is more than max length  
        else:
            headline = headline[:max_headline_length]
            for word in headline:
                pad_daily_headlines.append(word)
    
    # Pad daily_headlines if they are less than max length
    if len(pad_daily_headlines) < max_daily_length:
        for i in range(max_daily_length-len(pad_daily_headlines)):
            pad = vocab_to_int["<PAD>"]
            pad_daily_headlines.append(pad)
    # Limit daily_headlines if they are more than max length
    else:
        pad_daily_headlines = pad_daily_headlines[:max_daily_length]
    pad_headlines.append(pad_daily_headlines)


# In[304]:

# Normalize opening prices (target values)
max_price = max(price)
min_price = min(price)
mean_price = np.mean(price)
std_price = np.std(price)
def normalize(price):
    return (price - mean_price)/std_price


# In[305]:

norm_price = []
for p in price:
    norm_price.append(normalize(p))

# Check that normalization worked well
print(min(norm_price))
print(max(norm_price))
print(np.mean(norm_price))


x_train, x_test, y_train, y_test = train_test_split(pad_headlines, norm_price, test_size = 0.15, random_state = 2)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("SAVING ARRAYS")



"""

import dill                            
filename = '../data/globalsave.pkl'
#dill.dump_session(filename)  #Save session
dill.load_session(filename) # load the session

dj = pd.read_csv("../data/DowJones.csv") #read in stock prices
news = pd.read_csv("../data/News.csv") #read in news data
dj.head()
dj.isnull().sum()
news.isnull().sum()
news.head()

# Calculate the difference in opening prices between the following and current day.
# The model will try to predict how much the Open value will change beased on the news.
dj = dj.set_index('Date').diff(periods=1)
dj['Date'] = dj.index
dj = dj.reset_index(drop=True)

dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1) # Remove unneeded features from the dataset

dj.head()
dj = dj[dj.Open.notnull()] # Remove top row since it has a null value.
dj.isnull().sum() # Check if there are any more null values.

# Create a list of the opening prices and their corresponding daily headlines from the news
price = []
headlines = []
dataset = np.zeros((1988,1))
x = 0
x = int(x)
for row in dj.iterrows():
    daily_headlines = []
    number = (row[1]['Open'])
    dataset[x]= number
    #dataset[x][1]= int(x)
    x=int(x)+1
    
print(dataset)

# Create lookback_array 
lookback = 5
lookback_array = np.zeros((dataset.shape[0],lookback))
for row in range(dataset.shape[0]):
 #   print(row)
    if 1 <= row < lookback:
        lookback_array[row][0:row] = dataset[0:row].T

    elif row >= lookback:
        lookback_array[row][0:] = dataset[(row-lookback):row].T

# Split lookbback_array for x2_test and x2_train
x2_test = lookback_array[1689:][:] 
x2_train = lookback_array[0:1689][:]


std_price = np.std(dataset,axis=0)
print("Mean Price")
print(mean_price)
print("STD Price")
print(std_price)



y_train = np.zeros(x2_train.shape[0])
y_test = np.zeros(x2_test.shape[0])
y_train = dataset[0:1689]
y_test = dataset[1689:]

#Normalization
x2_test = (x2_test - mean_price)/std_price
x2_train = (x2_train - mean_price)/std_price
y_test = (y_test - mean_price)/std_price
y_train = (y_train - mean_price)/std_price



print("shape of x2_test: " + str(x2_test.shape))
print("shape of x2_train: " + str(x2_train.shape))
print("shape of x_train: "+ str(x_train.shape))
print("shape of y_train"+str(y_train.shape))
print("shape of x_test"+str(x_test.shape))
print("shape of y_test"+str(y_test.shape))
print("contents of y_tetst: "+ str(y_test))
print("contents of y_tetst: "+ str(y_train))



x2_train = np.reshape(x2_train,(x2_train.shape[0],1,x2_train.shape[1]))
x2_test = np.reshape(x2_test,(x2_test.shape[0],1,x2_test.shape[1]))

filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
other_weights = initializers.glorot_normal(seed = 5)
nb_filter = 16
rnn_output_size = 128
hidden_dims = 128
wider = True
deeper = True

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2




def build_model():
    
    cnn1 = Sequential()
    cnn1.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))

    cnn1.add(Dropout(dropout)) 

    cnn1.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length1, 
                             padding = 'same',
                            activation = 'relu'))

    cnn1.add(Dropout(dropout))

    if deeper == True:
        cnn1.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length1, 
                                 padding = 'same',
                                activation = 'relu'))
        cnn1.add(Dropout(dropout))

    cnn1.add(LSTM(rnn_output_size, 
                   activation=None,
                   kernel_initializer=weights,
                   dropout = dropout))

    

    cnn2 = Sequential()
    cnn2.add(Embedding(nb_words, 
                         embedding_dim,
                         weights=[word_embedding_matrix], 
                         input_length=max_daily_length))
    cnn2.add(Dropout(dropout))
    cnn2.add(Convolution1D(filters = nb_filter, 
                             kernel_size = filter_length2, 
                             padding = 'same',
                             activation = 'relu'))
    cnn2.add(Dropout(dropout))
    
    if deeper == True:
        cnn2.add(Convolution1D(filters = nb_filter, 
                                 kernel_size = filter_length2, 
                                 padding = 'same',
                                 activation = 'relu'))
        cnn2.add(Dropout(dropout))
    
    cnn2.add(LSTM(rnn_output_size, 
                    activation=None,
                    kernel_initializer=weights,
                    dropout = dropout))
    

    djlstm=Sequential()
    djlstm.add(LSTM(4, input_shape=(1, lookback), kernel_initializer = other_weights,activation='relu'))

    full_conn = Sequential()
 
    full_conn = Add()([cnn1.output,cnn2.output])
    full_conn = Concatenate()([full_conn, djlstm.output])
    full_conn = Dense(hidden_dims, kernel_initializer=other_weights)(full_conn)
    full_conn = Dropout(dropout)(full_conn)
    
    if deeper == True:
        full_conn = Dense(hidden_dims//2, kernel_initializer=weights)(full_conn)
        full_conn = Dropout(dropout)(full_conn)

    full_conn = Dense(1,kernel_initializer = other_weights, name='output')(full_conn)

    complete_model = Model(inputs=[cnn1.input, cnn2.input, djlstm.input], outputs=full_conn)
    complete_model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate,clipvalue=1.0))
    print(complete_model)
    return complete_model




max_count = 1




from random import randint

best_val = np.inf
best_lr = 0
best_deep = 0
best_wide = 0
best_dropout = 0



for count in range(max_count):
    deeper = bool(randint(0,1))
    wider = bool(randint(0,1))
    dropout = round(np.random.uniform(0.3,0.5),2)
    lr_p = -1 * np.random.rand() -3
    lr = 10 ** lr_p
    lr = round(lr,5)

    model = build_model()
    print()
    print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
    print()
    
    # save_weights = '../weights/deeper={}_wider={}_lr={}_dropout={}.h5'.format(
    #                 deeper,wider,lr,dropout)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]
    print('model: ' + str(model))

    history = model.fit([x_train,x_train, x2_train],
                                    y_train,
                                    batch_size=256,
                                    epochs=100,
                                    validation_split=0.15,
                                    verbose=True,
                                    shuffle=True,
                                    callbacks = callbacks)
    
    print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))

    if (history.history['loss'])[-1] < best_val:
        best_model = model
        best_val = (history.history['loss'])[-1]
        best_deep = deeper
        best_wide = wider
        best_lr = lr
        best_dropout = dropout
        print("SAVING BESt")
        save_best_weights = '../weights/deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    best_deep,best_wide,best_lr,best_dropout)
        best_model.save(save_best_weights)

# Make predictions with the best weights



# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()

model.load_weights('../weights/deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    best_deep,best_wide,best_lr,best_dropout))
#model.load_weights('./question_pairs_weights_deeper=True_wider=True_lr=0.001_dropout=0.3.h5')


predictions = model.predict([x_test,x_test,x2_test], verbose = True)

# predictions = model.predict([x_test, x_test, x2_test])
model_show_predictions(predictions, y_test, deeper, wider, dropout, 
                    learning_rate, std_price=std_price, mean_price=mean_price)



