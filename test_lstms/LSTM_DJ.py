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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math as math
nltk.download('stopwords')


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

np.random.seed(7)

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


print("DATASET")
print(type(dataset))


import numpy

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print("MORE SHAPES: ")
print(trainX.shape)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
dropout=0.5
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))


# model.add(Dense(8))
# model.add(Dropout(dropout))
# model.add(Activation('relu'))
# model.add(Dense(2))


model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


print("SHAPES")
print(trainPredict.shape)
print(testPredict.shape)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

print(testY.shape)
print(testPredict.shape)

direction_pred = []
for pred in testPredict:
    if pred >= 0:
        direction_pred.append(1)
    else:
        direction_pred.append(0)
direction_test = []
for value in testY[0]:
    if value >= 0:
        direction_test.append(1)
    else:
        direction_test.append(0)




# Calculate if the predicted direction matched the actual direction
direction = acc(direction_test, direction_pred)
direction = round(direction,4)*100
print("Predicted values matched the actual direction {}% of the time.".format(direction))





#plt.show()]
plt.savefig('LSTM_PLOT_YEET.png')

print(history.history.keys())
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('../results/oooyea2.png')


