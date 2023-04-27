

# Commented out IPython magic to ensure Python compatibility.

# Import packages

import numpy as np
import pandas as pd
import chardet
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import string
import time

# Display plots inline
# %matplotlib inline

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.callbacks import ModelCheckpoint

from scipy import sparse, stats

# Download nltk - only need to run once
nltk.download('stopwords')

# Get encoding of data file
# Get encoding of data file
with open("C:/Users/Karthik Vaddi/Downloads/author_data.csv", 'rb') as file:
    print(chardet.detect(file.read()))

# Load data (uncomment relevant line)
# Local version
#data = pd.read_csv("author_data.csv", encoding="Windows-1252")

# Floydhub version
data = pd.read_csv("C:/Users/Karthik Vaddi/Downloads/author_data.csv", encoding="utf-8")
print(data.head())
# Create feature (text) and label (author) lists
text = list(data['text'].values)
author= list(data['author'].values)

print("The author dataset contains {} datapoints.".format(len(text)))

# Create list of accented characters
accented_chars = ['ï', 'é', 'ñ', 'è', 'ö', 'æ', 'ô', 'â', 'á', 'à', 'ê', 'ë', '€', 'œ', '€™', '€˜', '*', '/', '{', '}',
                  'Ã', 'Â']

# Find all texts containing unusual characters
accented_text = []

for i in range(len(text)):
    for j in text[i]:
        if j in accented_chars:
            accented_text.append(i)

accented_text = list(set(accented_text))

print('There are', str(len(accented_text)), 'texts containing accented characters.')

# Remove invalid character from text
text = [excerpt.replace('\xa0', '') for excerpt in text]

# Verify character has been removed
unusual_text = []

for i in range(len(text)):
    for j in text[i]:
        if j == accented_chars:
            unusual_text.append(i)

unusual_text = list(set(unusual_text))

print('There are', str(len(unusual_text)), 'texts containing the invalid character.')

# Remove blocks of white space
new_text = []

for excerpt in text:
    while "  " in excerpt:
        excerpt = excerpt.replace("  ", " ")
    new_text.append(excerpt)

text = new_text
print(len(text))

from sklearn.model_selection import train_test_split

normed_text = []

for i in range(len(text)):
    new = text[i].lower()
    new = new.translate(str.maketrans('', '', string.punctuation))
    new = new.replace('“', '').replace('”', '')
    normed_text.append(new)

print(normed_text[0:5])
print(len(normed_text))

text_train, text_test, author_train, author_test = train_test_split(normed_text, author, test_size=0.2)

# Check shapes of created datasets

print(np.shape(text_train))
print(np.shape(text_test))
print(np.shape(author_train))
print(np.shape(author_test))

import nltk
from nltk.util import ngrams

nltk.download('punkt')

from collections import Counter
import numpy as np
import nltk
import nltk
import itertools
from collections import Counter
import itertools

nltk.download('punkt')
text_train, text_test, author_train,author_test = train_test_split(normed_text, author, test_size = 0.2, random_state = 5)
# generating vocab size similar to def get_vocab_size(excerpt_list, n, seq_size)
def generate_word_ngrams(normed_text,n):
    word_ngrams = []
    n_gram_list = []
    seq_size = 350
    #   print(normed_text)
    for sentence in normed_text:
        tokens = nltk.word_tokenize(sentence)

        # Filter out punctuation and lowercase words
        filtered_tokens = [token.lower() for token in tokens if token.isalpha()]

        # Generate n-grams from filtered tokens
        ngrams_list = list(zip(*[filtered_tokens[i:] for i in range(n)]))
        print("the ngrams _list ")
        print(ngrams_list)
        #     print(len(ngrams_list))
        gram_len = len(ngrams_list)

        if gram_len >= seq_size:
            ngrams_list = ngrams_list[0:seq_size]
        else:
            diff = seq_size - gram_len
            extra = [0] * diff
            ngrams_list = ngrams_list + extra
            print(ngrams_list)
            print(len(ngrams_list))
        n_gram_list.append(ngrams_list)
   # print("before flattended")
    # Flatten n-gram list
    lists_list = [[elem for elem in tup] for tup in n_gram_list]
   # print(lists_list)
    lists_list = list(np.array(lists_list).flat)

    # Calculate vocab size
    n_gram_cnt = Counter(lists_list)
    vocab_size = len(n_gram_cnt)

  #  print("the vocab size is \n")


    return vocab_size


# Example usage

generate_word_ngrams(normed_text[0:5],3)



def create_n_grams(excerpt_list, n, vocab_size, seq_size):
    """Create a list of n-gram sequences

    Args:
    excerpt_list: list of strings. List of normalized text excerpts.
    n: int. Length of n-grams.
    vocab_size: int. Size of n-gram vocab (used in one-hot encoding)
    seq_size: int. Size of n-gram sequences

    Returns:
    n_gram_array: array. Numpy array of one-hot encoded n-grams.
    """
    n_gram_list = []

    for excerpt in excerpt_list:
        # Remove spaces
        excerpt = excerpt.replace(" ", "")

        # Extract n-grams
        n_grams = [excerpt[i:i + n] for i in range(len(excerpt) - n + 1)]

        # Convert to a single string with spaces between n-grams
        new_string = " ".join(n_grams)

        # One hot encode
        hot = one_hot(new_string, round(vocab_size * 1.3))

        # Pad hot if necessary
        hot_len = len(hot)
        if hot_len >= seq_size:
            hot = hot[0:seq_size]
        else:
            diff = seq_size - hot_len
            extra = [0] * diff
            hot = hot + extra

        n_gram_list.append(hot)

    n_gram_array = np.array(n_gram_list)
    print(n_gram_array)

    return n_gram_array
"""
for i in range(1,4):
    vocab_size = generate_word_ngrams(text_train, i)
    print('Vocab size for n =', i, 'is:', vocab_size)

"""

# Create n-gram lists


gram1_train = create_n_grams(text_train, 1, 39759, 350)
gram2_train = create_n_grams(text_train, 2, 407771, 350)
gram3_train = create_n_grams(text_train, 3, 880608, 350)
gram4_train = create_n_grams(text_train, 4, 1053567, 350)

gram1_test = create_n_grams(text_test, 1, 39759, 350)
gram2_test = create_n_grams(text_test, 2, 407771, 350)
gram3_test = create_n_grams(text_test, 3, 880608, 350)
gram4_test = create_n_grams(text_test, 4, 1053567, 350)

max_1gram = np.max(gram1_train)
max_2gram = np.max(gram2_train)
max_3gram = np.max(gram3_train)
max_4gram = np.max(gram4_train)

print('Maximum encoding value for 1-grams is: ', max_1gram)
print('Maximum encoding value for 2-grams is: ', max_2gram)
print('Maximum encoding value for 3-grams is: ', max_3gram)
print('Maximum encoding value for 4-grams is: ', max_4gram)

author_lb = LabelBinarizer()

author_lb.fit(author_train)
author_train_hot = author_lb.transform(author_train)
author_test_hot = author_lb.transform(author_test)
# Define model architecture in keras
# Code reference: https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
def define_model(input_len, output_size, vocab_size, embedding_dim, verbose=True,
                 drop_out_pct=0.25, conv_filters=500, activation_fn='relu', pool_size=2, learning=0.0001):
    """Define n-gram CNN

    Args:
    input_len: int. Length of input sequences.
    output_size: int. Number of output classes.
    vocab_size: int. Maximum value of n-gram encoding.
    embedding_dim: int. Size of embedding layer.
    verbose: bool. Whether or not to print model summary.
    drop_out_pct: float. Drop-out rate.
    conv_filters: int. Number of filters in the conv layer.
    activation_fn: string. Activation function to use in the convolutional layer.
    pool_size: int. Pool size for the max pooling layer.
    learning: float. Learning rate for the model optimizer.

    Returns:
    model: keras model object.
    """
    # Channel 1
    inputs1 = Input(shape=(input_len,))
    embedding1 = Embedding(vocab_size, embedding_dim)(inputs1)
    drop1 = Dropout(drop_out_pct)(embedding1)
    conv1 = Conv1D(filters=conv_filters, kernel_size=3, activation=activation_fn)(drop1)
    pool1 = MaxPooling1D(pool_size=pool_size)(conv1)
    flat1 = Flatten()(pool1)

    # Channel 2
    inputs2 = Input(shape=(input_len,))
    embedding2 = Embedding(vocab_size, embedding_dim)(inputs2)
    drop2 = Dropout(drop_out_pct)(embedding2)
    conv2 = Conv1D(filters=conv_filters, kernel_size=4, activation=activation_fn)(drop2)
    pool2 = MaxPooling1D(pool_size=pool_size)(conv2)
    flat2 = Flatten()(pool2)

    # Channel 3
    inputs3 = Input(shape=(input_len,))
    embedding3 = Embedding(vocab_size, embedding_dim)(inputs3)
    drop3 = Dropout(drop_out_pct)(embedding3)
    conv3 = Conv1D(filters=conv_filters, kernel_size=5, activation=activation_fn)(drop3)
    pool3 = MaxPooling1D(pool_size=pool_size)(conv3)
    flat3 = Flatten()(pool3)

    # Merge channels
    merged = concatenate([flat1, flat2, flat3])

    # Create output layer
    output = Dense(output_size, activation='softmax')(merged)

    # Create model
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning), metrics=['accuracy'])

    if verbose:
        print(model.summary())

    return model
#
#
# # Create the 1-gram model
#
# # gram1_model = define_model(350, 9, max_1gram + 1, 26)
# # print("the gram1 _model")
# # print(gram1_model)
# # # Train 1-gram CNN
# # gram1_model.fit([gram1_train, gram1_train, gram1_train], author_train_hot, epochs=7, batch_size=32,
# #               verbose=1, validation_split=0.2)
#
# # Create the 2-gram model
# # print("the gram2_model")
# # gram2_model = define_model(350, 9, max_2gram + 1, 100)
# #Train 2-gram CNN
# # gram2_model.fit([gram2_train, gram2_train, gram2_train], author_train_hot, epochs=7, batch_size=32,
# #                verbose=1, validation_split=0.2)
# # Create the 3-gram model
# print("the gram3_model")
# gram3_model = define_model(350, 9, max_3gram + 1, 100)
#
# # Train 3-gram CNN
#
# gram3_model.fit([gram3_train, gram3_train, gram3_train], author_train_hot, epochs=7, batch_size=32,
#                verbose=1, validation_split=0.2)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
    cm: matrix. Confusion matrix for plotting.
    classes: list. List of class labels.
    normalize: bool. Whether or not to normalize the confusion matrix.
    title: string. Title for plot.
    cmap: color map. Color scheme for plot.

    Returns:
    None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    df_cm = pd.DataFrame(cm, index=classes,
                         columns=classes)
    sns.heatmap(df_cm, annot=True, cmap=cmap)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)


# Fit and evaluate Model 1 (3-gram CNN)

t0 = time.time()

# Fit model
model1 = define_model(350, 9, max_3gram + 1, 100)
model1.fit([gram3_train, gram3_train, gram3_train], author_train_hot, epochs=7, batch_size=32,
           verbose = 1, validation_split = 0.2)
t1 = time.time()

# Predict values for test set
author_pred1 = model1.predict([gram3_test, gram3_test, gram3_test])

t2 = time.time()

# Reverse one-hot encoding of labels
# Reverse one-hot encoding of labels
author_pred1 = author_lb.inverse_transform(author_pred1)

# Evaluate
accuracy = accuracy_score(author_test, author_pred1)
precision, recall, f1, support = score(author_test,author_pred1)
ave_precision = np.average(precision, weights = support/np.sum(support))
ave_recall = np.average(recall, weights = support/np.sum(support))
ave_f1 = np.average(f1, weights = support/np.sum(support))
confusion = confusion_matrix(author_test, author_pred1, labels = ['Bram Stoker', 'Charles Dickens', 'Jane Austen', 'Jonathan Swift', 'Mark Twain', 'Oscar Wilde', 'Robert Louis Stevenson', 'Rudyard Kipling', 'Williams Shakespeare'])
print("Accuracy:", accuracy)
print("Ave. Precision:", ave_precision)
print("Ave. Recall:", ave_recall)
print("Ave. F1 Score:", ave_f1)
print("Training Time:", (t1 - t0), "seconds")
print("Prediction Time:", (t2 - t1), "seconds")
print("Confusion Matrix:\n", confusion)
# Plot normalized confusion matrix
plot_confusion_matrix(confusion, classes=[], \
                      normalize=True, title='Normalized Confusion Matrix - Model 1')

plt.savefig("confusion1.eps")