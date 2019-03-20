# cs230project
# Using Daily News to Predict the Stock Market

```
Jason N. Kurohara
Computer Science
Stanford University
jkuro@stanford.edu
```
```
Joshua R. Chang
Computer Science
Stanford University
jrchang@stanford.edu
```
```
Callan A. Hoskins
Computer Science
Stanford University
chosk@stanford.edu
```
## Abstract

```
Financial markets are inherently volatile and reflect diverse macroeconomic and
microeconomic trends. In this research project, we use natural language processing
in conjunction with a deep neural network to predict the daily change in price of
the Dow Jones Industrial Average. Our model was slightly better than random
guessing for predicting whether the stock market would rise and fall on a given
day.
```
## 1 Introduction

Much of the world’s wealth is located in large financial markets, which allow individuals and
organizations to purchase shares of companies on a public exchange. The price of a company’s shares
is subject to a wide range of variables–some intuitive and some not. The task of using these inputs to
predict a stock’s price movement is a trillion-dollar industry.

For decades, only large financial corporations and expert analysts have had access to the data
required for understanding financial markets. In the past decade, however, publicly-provided market
information has enabled independent investors to make much more intelligent, informed investment
decisions.

For most of the history of the stock market, investment decisions have been made by educated and
informed humans. Though the stock market has displayed a consistent overall trend for the duration
of its existence, results of attempts to ’beat’ the stock market have been mixed. It is unclear which
variables influence stock prices and by how much. Microeconomic trends and consumer preferences
often determine the price of a single stock, but these factors can be extremely difficult to predict; even
experts sometimes perform this task very poorly.

For this research project, we focus on predicting the price of the Dow Jones Industrial Average
(DJIA), a price-weighted index of thirty large American companies that represents the state of the
domestic economy as a whole. Since significant changes in the DJIA are caused by factors that affect
each of its constituent stocks, its price reflects macroeconomic trends. Previous research has shown
that macroeconomic trends can be predicted based on daily news, and further research shows that
news headlines alone can be used to achieve the same results.

We have constructed a neural network that predicts the float amount by which the DJIA rises or falls
based on a textual input consisting of twenty-five news headlines concatenated into one string per
day.

CS230: Deep Learning, Winter 2018, Stanford University, CA. (LateX template borrowed from NIPS 2017.)


## 2 Related work

Past work has used text drawn from various sources to generate predictions about stock market
movement. One team used Twitter [ 1 ] to extract public opinion and was able to use a neural network
to transform these opinions into rather accurate predictions about the DJIA closing price. Other
leading researchers have experimented with event detection algorithms [ 2 ] and other natural language
processing techniques to create rich input for neural networks that aim to learn their relationship to
stock prices.

Others have focused on developing a neural network architecture that is best able to convert these
inputs into accurate predictions. Since macroeconomic trends are closely tied to public opinion
and sentiment, effective sentiment analysis is key for our objective. One promising model uses a
convolutional neural network in conjunction with a recurrent neural network to conduct sentiment
analysis of short texts [ 3 ]. This work is applicable to our objective because it extracts sentiment from
documents that are too brief to contain much context, like news headlines.

## 3 Dataset and Features

We used a publicly available dataset of DJIA prices and daily news articles available from Kaggle
[ 4 ]. The DJIA data included the opening and closing price of the DJIA, plus other metrics such as its
daily trading volume, daily high, and daily low, for every market day from August 8, 2008 through
July 1, 2016. This dataset can also be downloaded from Yahoo Finance.

```
Figure 1: This is an example of all the metrics included for one day in our dataset.
```
The news data included the 25 most popular news headlines for every day (including weekends)
in the same range, sourced from Reddit News. The 25 news headlines for each day are those that
received the most votes from Reddit users.

For example, corresponding to the price change on July 1, 2016, there are 25 news headlines ranging
from "The president of France says if Brexit won, so can Donald Trump" to "Spain arrests three
Pakistanis accused of promoting militancy".

Our data included 10% of our total data, and our dev set included 10% of our total data, and our train
data included 80% of our total data.

Another key feature to each day in our dataset is the previous 5 days’ stock price changes. The
intuition behind this feature is that stock prices depend not only on the daily news headlines, but
also recent previous stock price movements. The number of previous stock price changes is a
hyperparameter that we tuned appropriately.‘ Thus, we implemented an LSTM to create a sequential
model in addition to newsheadlines over time. Our workflow and architecutre is explained in the
Methods section of this paper.

For our news input data, we clean the data before training our model. To do so, we convert the
headlines to lower case, replace contractions, remove unwanted characters, and remove stop words
that are irrelevant for this natural-language processing task. Next, we create embeddings, which are
low-dimensional feature vectors representing discrete data like strings or in this case, news headlines.
We use pre-trained, 300-dimensional GloVe embeddings for all words with corresponding GloVe
embeddings, and use random embeddings of the same size for words not found in GloVe’s vocabulary.
The GloVe vocabulary is used to create a vector space representation of words to capture semantic
and syntactic meaning. For more information on GloVe, see [5].

To structure our previous 5 days’ stock price changes, we create a two dimensional array with rows
that represent each day in our dataset. Every row then contains the previous 5 days’ stock price
changes. Thus, we end up with a (1988,5) NumPy array. For the first 5 days, we zero-pad the
non-existent entries. For instance, on July 1, 2016, the first row of this input data would be


Figure 2: This is a sample of the first 5 rows of our two-dimensional array. The first row corresponds
to the previous stock price changes on August 8, 2008 which is all zeros since this is the beginning of
our dataset. The sixth row corresponds to August 13, 2008 which contains the five previous stock
price changes.

## 4 Methods

The inputs to our neural network are sequential. More specifically, every day from August 8, 2008
to July 1, 2016 has 25 news headlines. Additionally, each input vector also contains the difference
between the DJIA stock price at timetand the DJIA stock price att− 1. In this model, we are not
predicting the actual stock price; instead, we predict the amount that stock price will rise or fall for
each day (a positive or negative difference). We also normalize the data by subtracting the mean and
dividing by the standard deviation of the stock price changes.

As our baseline, we simplified this task into a pretty straightforward sentiment analysis task. Because
we are working with brief news headlines, we used Wang et al’s approach to sentiment analysis [ 3 ],
feeding sentence embeddings into a CNN that takes the local features and inputs these results into a
RNN for sentiment analysis of short texts or news headlines. The output of the RNN is fed into a
fully-connected network, which aims to learn its relationship to sentiment. The model architecture
is shown inFigure 2. We used the mean squared error as our loss function and Adam optimization
algorithm. This network is fully implemented using Keras.

```
Figure 3: Model architecture
```
In this architecture, we apply windows and convolutional operations with different matrices in the
CNN while maintaining sequential information in the texts. Then, the RNN takes in these encoded
features and learns long-term dependencies.

We found an implementation of this network on Github [6] and were able to use it as our base.

More specifically, for our RNN, we use a Long Short-Term Memory (LSTM) that learns long-term
dependencies. Rather than using a traditional feed-forward network, an LSTM detects features from
an input sequence and carries the information over a long distance. Our model is built on the Keras
[7] and scikit-learn [8] machine learning frameworks.

In addition to the sentiment analysis on the textual inputs, we also included the change in price of the
stock market for the previous twenty days. We feed the previous twenty days’ price changes into
an LSTM, which then becomes another input to the fully connected network. This should help our
model recognize long-term trends in the DJIA’s price.


## 5 Experiments

Our baseline model relied on news data as the sole source of input. With this model and the optimal
hyperparameters found by [ 6 ], we were able to correctly predict whether the stock market would
rise/fall 54% of the time. This was our starting point. From this baseline, we iterated through several
experiments to develop our final model.

Our hyperparameters include different learning rates, the value of p in dropout, the number of layers
to include in the fully-connected layer, the number of elements in the LSTM, and the number of
convolution layers on the textual dta. We used both grid search and random search to find the best
possible values of these hyperparameters.

In an attempt to put the performance of our model in perspective with other datasets, we created a
simple logistic regression model that uses NYC weather data to predict whether the price of the DJIA
rises or falls. With this simple (and logically unrelated) data, we were able to predict positive/negative
price movements in the DJIA with 52% accuracy–slightly better than flipping a coin.

Figure 4: We tried 6 different models, varying the input data (weather, news, and previous stock
prices) and the model architecture. Our best accuracy results came from the deepest and widest
networks, which used more convolutional and fully connected layers.

## 6 Results

Using the previous stock price changes and global news headlines, we achieved 55.28% accuracy
with a mean absolute error of 71.40, meaning that this model more accurately predicted price changes
than our baseline.

Further tuning our hyperparameters, we increased the depth and width of our network by using 2 FC
layers, 2 convolutional layers, and 128 hidden dimensions in the fully connected network to further
increase our accuracy, reaching 58.28%.

Our best model used 4 fully connected layers, 4 1D convolutions for the textual news data , and 126
hidden dimensions in the fully connected layer, achieving 61.31% accuracy in predicting a positive or
negative change in the DJIA, a 7.03% increase of our baseline model. Our model was well-fit to the
data, since the training error for our our best model was around 58% while our test accuracy was
60.31%.


Figure 5: By varying the number of layers and hidden dimensions, we tested our models and observed
their validation loss. After training on less than 20 epochs using early stopping, our validation loss
still remained relatively high.

One notable trend throughout our experiments is the reasonably high mean absolute error, which
provides insight on the average magnitude of error in our predictions regardless of direction. Our
model is best suited for a binary classification task of predicting a rise/fall and struggles with
determining the magnitude of stock price changes.

## 7 Conclusion

Our model was able to outperform our baseline at the task of predicting whether the DJIA would rise
or fall each day. We started at 54% accuracy and eventually got to 61.31%–a marginal difference and
still barely better than a coin flip. This model didn’t completely succeed at its task of predicting the
overall stock price movements. However, it did far better than even many human analysts can do.

It was promising that our model reacted positively to the addition of historical market price data, as
that shows that the basic model with a textual news input can be augmented and bolstered by building
in relevant alternate input sources. This makes sense because the stock market is subject to many
factors, any of which can cause the market’s price to swing wildly. Historical price data is just one,
and there are many others we could incorporate into our model. Some interesting ideas include tweets
from the presidential Twitter account and price information from foreign markets. It could also be
interesting to apply this model on a company-specific basis, feeding it news headlines focused on a
single company. We suspect that this data would be even more relevant because it is directly relevant
to the health of the company (which should be reflected in its stock price).

## 8 Contributions

Josh: Callan:

## References

[1]Xiaojun Zeng Johan Bollen, Huina Mao. Twitter mood predicts the stock market.Journal of
Computational Science, 2(1):1–8, 2011.


[2]Ting Liu Junwen Duan Xiao Ding, Yue Zhang. Deep learning for event-driven stock prediction.
2014.

[3]Zhiyoung Luo Xingyou Wang, Weijie Jiang. Combination of convolutional and recurrent neural
network for sentiment analysis of short texts.COLING, 2016.

[4] Aaron7sun. Daily news for stock market prediction, 2016.

[5]Christopher D. Manning Jeffrey Pennington, Richard Socher. Glove: Global vectors for word
representation, 2014.

[6]David Currie. Predicting-the-dow-jones-with-headlines.https://github.com/Currie32/
Predicting-the-Dow-Jones-with-Headlines.

[7] Open Source Francois Challet. Keras deep learning framework.

[8] Open Source. scikit-learn machine learning framework.

