
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
from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, merge, BatchNormalization, Flatten, Reshape, Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
from keras.layers import *
from keras.layers import add
import time
import nltk
nltk.download('stopwords')
from datetime import datetime
from keras import regularizers

dj = pd.read_csv("../data/DowJones.csv") #read in stock prices
weather = pd.read_csv("../data/nyc_weather.csv") #read in news data
dj.head()
dj.isnull().sum()
weather.isnull().sum()

def norm_date(date):
    return  datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d')

weather.DATE = weather.DATE.apply(norm_date)
print(weather.head())
print(dj.head())

print("Dow Jones Raw Data Shape: " + str(dj.shape))
print("Raw weather Data Shae: " + str(weather.shape))

# Compare the number of unique dates. We want matching values.
print("Number of Unique Dates in DJ data set: " + str(len(set(dj.Date))))
print("Number of Unique Dates in news headlines: " + str(len(set(weather.DATE))))

# Remove 'station' and 'name' columns from weather
weather = weather.drop(['STATION', 'NAME'], axis=1)
print(weather.head())

# Calculate the difference in opening prices between the following and current day.
# The model will try to predict how much the Open value will change beased on the news.
dj = dj.set_index('Date').diff(periods=1)
dj['Date'] = dj.index
dj = dj.reset_index(drop=True)

dj = dj.drop(['High','Low','Close','Volume','Adj Close'], 1) # Remove unneeded features from the dataset

dj.head()
dj = dj[dj.Open.notnull()] # Remove top row since it has a null value.
dj.isnull().sum() # Check if there are any more null values.

# Remove NaN values
weather = weather.dropna()
dj = dj.dropna()

# Remove the extra dates that are in news
weather = weather[weather.DATE.isin(dj.Date)]
dj = dj[dj.Date.isin(weather.DATE)]

print(weather.head())
print(weather['DATE'])
print(weather.columns)

print("Removed extra dates in DJ data set: " + str(len(set(dj.Date))))
print("Removed extra dates in DJ data set: " + str(len(set(weather.DATE))))
print("Weather HEAD")
print(weather.head())
print('DJ HEAD')
print(dj.head())

weather = weather.drop(['DATE'], axis=1)

# Create a list of the opening prices and their corresponding daily headlines from the news
price = []

for row in dj.iterrows():
    price.append(row[1]['Open'])
    
    # Track progress
    if len(price) % 500 == 0:
        print(len(price))


print("Length of prices: " + str(len(price))) # Compare lengths to ensure they are the same

# Normalize opening prices (target values)
max_price = max(price)
min_price = min(price)
mean_price = np.mean(price)
def normalize(price):
    return ((price-min_price)/(max_price-min_price))

norm_price = []
for p in price:
    norm_price.append(normalize(p))


# Check that normalization worked well
print(min(norm_price))
print(max(norm_price))
print(np.mean(norm_price))

print('weather has nulls: ' + str(weather.isnull().any()))
print('weather PRCP nulls: ')
print(weather[weather.PRCP.isnull()])
print('weather TMAX nulls: ')
print(weather[weather.TMAX.isnull()])
print('price has nulls: ' + str(pd.DataFrame(norm_price).isnull().any()))


####################################
print("shape of price: " + str(len(price)))
print("shape of price: " + str(np.array(price).shape))

x_train, x_dev, y_train, y_dev = train_test_split(weather, norm_price, test_size = 0.15, random_state = 2)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_dev = np.array(x_dev)
y_dev = np.array(y_dev)



filter_length1 = 3
filter_length2 = 5
dropout = 0.5
learning_rate = 0.001
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)
nb_filter = 16
rnn_output_size = 128
hidden_dims = 128
wider = True
deeper = True
# weather = np.transpose(weather)
input_shape = x_train.shape
print('input_shape: ' + str(input_shape))
n_cols = input_shape[1]

if wider == True:
    nb_filter *= 2
    rnn_output_size *= 2
    hidden_dims *= 2


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='relu', input_shape=(n_cols,), 
        kernel_regularizer=regularizers.l2(0.01), 
        activity_regularizer=regularizers.l1(0.01), 
        kernel_initializer=weights))
    return model


# Use grid search to help find a better model
for deeper in [False]:
    for wider in [True,False]:
        for learning_rate in [0.001]:
            for dropout in [0.3, 0.5]:
                model = build_model()
                print()
                print("Current model: Deeper={}, Wider={}, LR={}, Dropout={}".format(
                    deeper,wider,learning_rate,dropout))
                print()
                save_best_weights = 'question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(
                    deeper,wider,learning_rate,dropout)
                callbacks = [ModelCheckpoint(save_best_weights, monitor='loss', save_best_only=False),
                             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
                             ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=3)]
                print('model: ' + str(model))
                opt = Adam()
                model.compile(optimizer=opt, loss='mse', metrics = ['mean_squared_error'])
                history = model.fit(x_train,
                                    y_train,
                                    epochs=100,
                                    batch_size=64,
                                    callbacks = callbacks)


# In[312]:

# Make predictions with the best weights
deeper=False
wider=False
dropout=0.3
learning_Rate = 0.001
# Need to rebuild model in case it is different from the model that was trained most recently.
model = build_model()

model.load_weights('./question_pairs_weights_deeper={}_wider={}_lr={}_dropout={}.h5'.format(deeper,wider,learning_rate,dropout))
predictions = model.predict([x_dev], verbose = True)


# In[314]:

def unnormalize(price):
    '''Revert values to their unnormalized amounts'''
    price = price*(max_price-min_price)+min_price
    return(price)


# In[345]:

unnorm_predictions = []
for pred in predictions:
    unnorm_predictions.append(unnormalize(pred))
    
unnorm_y_test = []
for y in y_dev:
    unnorm_y_test.append(unnormalize(y))



# In[362]:

print("Summary of actual opening price changes")
print(pd.DataFrame(unnorm_y_test, columns=[""]).describe())
print()
print("Summary of predicted opening price changes")
print(pd.DataFrame(unnorm_predictions, columns=[""]).describe())


# In[365]:

# Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(12,4))
plt.plot(unnorm_predictions)
plt.plot(unnorm_y_test)
plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")


# In[318]:

# Create lists to measure if opening price increased or decreased
direction_pred = []
for pred in unnorm_predictions:
    if pred >= 0:
        direction_pred.append(1)
    else:
        direction_pred.append(0)
direction_test = []
for value in unnorm_y_test:
    if value >= 0:
        direction_test.append(1)
    else:
        direction_test.append(0)


#Total Training time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()
#print ("Average epoch training time: {} seconds".format(np.mean(time_callback.times)))

# Calculate errors
mae = mae(unnorm_y_test, unnorm_predictions) #median absolute error
rmse = np.sqrt(mse(y_dev, predictions)) # root mean squared error
r2 = r2(unnorm_y_test, unnorm_predictions) #R squared error

print("Median absolute error: {}".format(mae))
print("Root mean suqared error: {}".format(rmse))
print("R squared error: {}".format(r2))

# Calculate if the predicted direction matched the actual direction
direction = acc(direction_test, direction_pred)
direction = round(direction,4)*100
print("Predicted values matched the actual direction {}% of the time.".format(direction))

#Display the graph
plt.show()