import keras
import re
import pandas as pd
import numpy as np
import os

def filter_words(words):
    '''Filters stop words'''
    words = words.lower()
    stopList = set(readFile(os.path.join('data', 'english.stop')))
    filtered = []
    for word in words:
        if not word in stopList and word.strip() != '':
            filtered.append(word)
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
        # Replace contractions with their longer forms
        words = words.split()
        new_words = []
        for word in words:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = ' '.join(new_text)

        #format words and remove unwanted characters
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'0,0', '', text)
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
        
        return text



#Loads datasets into X_train, Y_train, X_dev, etc...
#After function call, X will contain embeddings of news articles with stop words
#removed and Y will contain the corresponding label (1 if the DJIA rose in
#price and 0 if it fell
def load_datasets():
    news = pd.read_csv(os.path.join('data', 'RedditNews.csv'))
    stocks = pd.read_csv(os.path.join('data', 'DJIA_table.csv'))
    
    #remove weekend news
    news = [news.Date.isin(stocks.Date)]
    
    #sets index as the 'Date' value for each row
    stocks = stocks.set_index('Date').diff(periods=1)
    #sets the 'Date' value for each row as the index value of each row
    stocks['Date'] = stocks.index
    #reset index to 0,1,2,... but drop the 'Date' column from the dataset
    stocks = stocks.reset_index(drop=True)

    #remove unnecessary features
    stocks = stocks.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)

    #remove top row with null value
    stocks = stocks[stocks.Open.notnull()]

    print('number of null values remaining: ' + str(stocks.isnull().sum()))


    prices = []
    headlines = []

    for row in stocks.iterrows():
        daily_headlines = []
        date = row[1]['Date']
        price.append(row[1]['Open'])
        for row_ in news[news.Date==date].iterrows():
            daily_headlines.append(row_[1]['News'])

        #Make sure pass goes successfully
        headlines.append(daily_headlines)
        if len(price) % 500 == 0:
            print(len(price))

    #Clean headlines
    clean_headlines = []
    for daily_headlines in headlines:
        clean_daily_headlines = []
        for headline in daily_headlines:
            clean_daily_headlines.append(clean_text(headline))
        clean_headlines.append(clean_daily_headlines)


    word_counts = {}
    for date in clean_headlines:
        for headline in date:
            for word in headline.split():
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts += 1

    print('Size of vocabulary: ' + str(len(word_counts)))

    #Load GloVe's embeddings
    embeddings_index = {}
    with open(os.join('data', 'glove.840B.300d.txt', encoding='utf-8')) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    print('Word embeddings: '  + str(len(embeddings_index)))

    #Find the number of words that are missing from GlOve, and
    #are used more than our threshold
    missing_words = 0
    threshold = 0

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1

    missing_ratio = round(missing_words/len(word_counts),4)*100

    print('Number of words missing from GloVe: ' + str(missing_words))
    print('Percent of words that are missing from vocab: ' + str(missing_ratio))

    vocab_to_int = {}

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1
    
    #special tokens that will be added to our vocab
    codes = ['<UNK>', '<PAD>']

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    #Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    print('Total number of unique words: ' + str(len(word_counts)))
    print('Number of words we will use: ' + str(len(vocab_to_int)))
    print('Percent of words we will use: ' + str(usage_ratio))

    embedding_dim = 300

    nb_words = len(vocab_to_int)
    # Create a matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            #if word not in GloVe, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding

    # Check if value matches len(vocab_to_int)
    print(len(word_embedding_matrix))

    # Change the text from words to integers
    # If word not in vocab, replace it with <UNK>
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
                    int_headline.append(vocab_to_int['<UNK>'])
                    unk_count += 1
            int_daily_headlines.append(int_headline)
        int_headlines.append(int_daily_headlines)

    #Find the length of headlines
    lengths = []
    for date in int_headlines:
        for headline in date:
            lengths.append(len(headline))

    #Create a dataframe so that the values can be inspected
    lengths = pd.DataFrame(lengths, columns=['counts'])

    lengths.describe()

    #Limit the length of a day's news to 200 words, and the length of any 
    #headline to 16 words
    #These values are chosen to not have an excessively long training
    #time and balance the number of headlines used and the number of words
    #from each headline.
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
                pad = vocab_to_int['<PAD>']
                pad_daily_headlines.append(pad)

        # Limit daily_headlines if they are not more than max lenth
        else:
            pad_daily_headlines = pad_daily_headlines[:max_daily_length]
        pad_headlines.append(pad_daily_headlines)

    # Normalize opening prices (target values)
    max_price = max(price)
    min_price = min(price)
    mean_price = np.mean(price)

    norm_price = []
    for p in price:
        norm_price.append(normalize(p))

    #Check that normalization worked well
    print(min(norm_price))
    print(max(norm_price))
    print(np.mean(norm_price))

    #Split data into training and testing sets
    # Validating data will be created during training. 
    x_train, x_test, y_train, y_test = train_test_split(pad_headlines, norm_price, 
            test_size = 0.15, random_state = 2)
    x_train = np.array(x_train)
    x_test_np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


def normalize(price):
    return ((price - min_price) / (max_price - min_price))






