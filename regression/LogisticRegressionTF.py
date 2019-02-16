import math
import os
import numpy as np
import h5py
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops

#Reads in a file as text
def readFile(fileName):
    contents = []
    f = open(fileName, encoding='latin-1')
    contents = f.read()
    f.close()
    return contents

#Removes English stop words from parameter 'words'
def filterStopWords(words):
    """Filters stop words."""
    stopList = set(readFile(os.path.join('data', 'english.stop')))
    filtered = []
    for word in words:
        if not word in stopList and word.strip() != '':
            filtered.append(word)
    return ' '.join(filtered)

#Loads datasets into X_train, Y_train, X_dev, etc...
#After function call, X will contain news articles with stop words removed
#and Y will contain the corresponding label(1 if the DJIA rose in price, 0 if it fell)
def load_datasets():
    X_train = []
    Y_train = []
    X_dev = []
    Y_dev = []
    X_test = []
    Y_test = []
    news = pd.read_csv(os.path.join('data', 'RedditNews.csv'))
    datasets = ['train','dev','test']
    #opportunity to speed up: avoid passing over entire news dataset three times
    for dataset in datasets:
        djiaPath = os.path.join('data', 'DJIA_table_' + dataset + '.csv')
        stocks = pd.read_csv(djiaPath)
        #binarize stock data
        stocks['Y'] = pd.Series(stocks['Close'] - stocks['Open'])
        stocks['Y'] = stocks['Y'] >= 0
        stocks['Y'] *= 1
        for row in range(news.shape[0]):
            if not row%1000:
                print(row)
            words = news.loc[row]['News']
            words = filterStopWords(words.split())
            #find stock price change corresponding to news headline
            stock_row = stocks.loc[stocks['Date']==news.loc[row]['Date']].index
            #if there is no matching date in the given stock data, skip that data point
            if len(stock_row) < 1:
                continue
            row_num = stock_row[0]
            klass = stocks.loc[row_num]['Y']
            if dataset == 'train':
                X_train.append(words)
                Y_train.append(klass)
            elif dataset == 'dev':
                X_dev.append(words)
                Y_dev.append(klass)
            else:
                X_test.append(words)
                Y_test.append(klass)
    print("X_train shape: " + str(len(X_train)))
    print("Y_train shape: " + str(len(Y_train)))
    print("X_dev shape: " + str(len(X_dev)))
    print('Y_dev shape: ' + str(len(Y_dev)))
    print('X_test shape: ' + str(len(X_test)))
    print('Y_test. shape: ' + str(len(Y_test)))
    #write down results - save time
    pd.Series(X_train).to_csv('X_train.csv')
    pd.Series(Y_train).to_csv('Y_train.csv')
    pd.Series(X_dev).to_csv('X_dev.csv')
    pd.Series(Y_dev).to_csv('Y_dev.csv')
    pd.Series(X_test).to_csv('X_test.csv')
    pd.Series(Y_test).to_csv('Y_test.csv')
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def main():
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_datasets()


#X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_datasets()

X_train = pd.read_csv("X_train.csv").values
Y_train = pd.read_csv("Y_train.csv").values
X_dev = pd.read_csv("X_dev.csv").values
Y_dev = pd.read_csv("Y_dev.csv").values
X_test = pd.read_csv("X_test.csv").values
Y_test = pd.read_csv("Y_test.csv").values

'''
All code from here below was cut from aymericdamien's Tensorflow-Examples on Github
or from stack exchange... and it doesn't work
'''

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# Set model weights
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.nn.sigmoid(tf.matmul(W, x) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-(y*tf.log(pred) + (1 - y)*tf.log(1 - pred)))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: X_train, y: Y_train})
        if epoch % 50 == 0:
            print('loss = %f' %(c))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_dev, y: Y_dev}))


if __name__ == '__main__':
    main()
