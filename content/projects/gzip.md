---
title: "K Nearest Neighbour Sentiment Analysis with Gzip Embeddings"
type: page
---

## Abstract
This code implements a K Nearest Neighbors (KNN) sentiment analysis model using normalized compression distance embeddings, applied to the [Sentiment & Emotions Labelled Tweets](https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets) dataset. The dataset consists of labeled tweets categorized into positive, neutral, and negative sentiments. The code preprocesses the data by cleaning text and converting sentiment labels into numerical values. It then splits the data into training and testing sets. A compression distance normalization method is employed to calculate the normalized compression distances between pairs of tweets. The training set's normalized compression distances are computed in parallel using multiprocessing for efficiency. The KNN model is trained using different numbers of neighbors (4, 5, 6, and 7), and their accuracies are evaluated on the test set. The model achieves accuracies ranging from approximately 51% to 60%, demonstrating the effectiveness of compression distances as features for sentiment analysis.

## Imports
``` python
import gzip
import time
import pickle
import multiprocessing
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

## Load Dataset
``` python
# There are a total of 24639 samples
n_samples = 5000

df = pd.read_csv('/kaggle/input/sentiment-and-emotions-of-tweets/sentiment-emotion-labelled_Dell_tweets.csv')
df = df.truncate(0, n_samples)
```

``` python
X = df['Text']
X.head()
```

![x](/images/gzip/x.png "x")

``` python
y = df['sentiment']
y.head()
```

![y](/images/gzip/y.png "y")


## Data Cleaning and Preparation
``` python
X = X.str.replace(r'@[^ ]+', '', regex=True) # Remove tagged users
X = X.str.replace(r'#[^ ]+', '', regex=True) # Remove hashtags
X = X.str.replace(r'http[^ ]+', '', regex=True) # Remove hashtags
X.head()
```

![xclean](/images/gzip/xclean.png "xclean")

``` python
y_labels = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

y = y.map(y_labels) # Map string labels to integers
y.head()
```

![yclean](/images/gzip/yclean.png "yclean")

## Train/Test splitting
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1, 
                                                    stratify=y)
```

## How to Normalize Compression Distances
``` python
X1_compressed = len(gzip.compress(X_train[0].encode()))
X1_compressed
```
> 177

``` python
X2_compressed = len(gzip.compress(X_train[1].encode()))
X2_compressed
```
> 121

``` python
XX = len(gzip.compress((" ".join([X_train[0], X_train[1]])).encode()))
XX
```
> 255

``` python
NCD = (XX - min(X1_compressed, X2_compressed)) / max(X1_compressed, X2_compressed)
NCD
```
> 0.7570621468926554

## Compression Distance Normalization
``` python
def calculate_ncd(x1, x2):
    X1_compressed = len(gzip.compress(x1.encode()))
    X2_compressed = len(gzip.compress(x2.encode()))  
    XX = len(gzip.compress((" ".join([x1, x2])).encode()))
  
    NCD = (XX - min(X1_compressed, X2_compressed)) / max(X1_compressed, X2_compressed)
    return NCD
```

``` python
def calculate_train_ncd(X_train):
   NCD = [[calculate_ncd(X_train.iloc[i], X_train.iloc[j]) for j in range(len(X_train))] for i in range(len(X_train))]
   return NCD

def calculate_test_ncd(X_test, X_train):
   NCD = [[calculate_ncd(X_test.iloc[i], X_train.iloc[j]) for j in range(len(X_train))] for i in range(len(X_test))]
   return NCD
```

``` python
CPU_CORES = multiprocessing.cpu_count()

with multiprocessing.Pool(CPU_CORES) as pool:
    train_NCD = pool.apply(calculate_train_ncd, [X_train])

with multiprocessing.Pool(CPU_CORES) as pool:
    test_NCD = pool.apply_async(calculate_test_ncd, args=(X_test, X_train))
    test_NCD = test_NCD.get()
```

## Training
``` python
# KNN classification
knn4 = KNeighborsClassifier(n_neighbors=4) 
knn4.fit(train_NCD, y_train)
knn5 = KNeighborsClassifier(n_neighbors=5) 
knn5.fit(train_NCD, y_train)
knn6 = KNeighborsClassifier(n_neighbors=6) 
knn6.fit(train_NCD, y_train)
knn7 = KNeighborsClassifier(n_neighbors=7) 
knn7.fit(train_NCD, y_train)
```

``` python
y_pred4 = knn4.predict(test_NCD)
y_pred5 = knn5.predict(test_NCD)
y_pred6 = knn6.predict(test_NCD)
y_pred7 = knn7.predict(test_NCD)


score4 = accuracy_score(y_test, y_pred4, normalize=True)
print('Accuracy: ', score4)
score5 = accuracy_score(y_test, y_pred5, normalize=True)
print('Accuracy: ', score5)
score6 = accuracy_score(y_test, y_pred6, normalize=True)
print('Accuracy: ', score6)
score7 = accuracy_score(y_test, y_pred7, normalize=True)
print('Accuracy: ', score7)
```

![accuracy](/images/gzip/accuracy.png "accuracy")