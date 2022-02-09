# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:20:03 2021

@author: Dipanwita
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import tensorflow.compat.v1 as v1
from sklearn.model_selection import LeaveOneOut
#import tensorflow_datasets as tfds
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
#import tensorflow.compat.v1 as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
import time
start_time = time.time()					#To keep track of time to run the code

print('Loading data ...')
		#Loading Accelerometer & Gyroscope data
data1 = pd.read_csv('D:/Research Program/phone_accelerometer.csv')
data2 = pd.read_csv('D:/Research Program/phone_gyroscope.csv')


length = len(data1)
data1 = data1.drop(labels = ['Arrival_Time','Creation_Time','Index', 'User'], axis=1)		#Dropping the unnecessary fields
data2 = data2.drop(labels = ['Arrival_Time','Creation_Time','Index', 'User','Model','Device'], axis=1)

data1 = data1.head(length)							#Taking only the top 'length' number of entries from both the data
data2 = data2.head(length)

data2.columns = ['x1', 'y1', 'z1', 'gt1']					#Renaming the column values of data2 as data1 would have same 'x','y' and 'z' variables

data = pd.concat([data1, data2], axis=1)			#Merging both the accelerometer and the gyroscope data			
to_drop = ['null']									#To drop the null values fro both data1 and data2
data = data[~data['gt'].isin(to_drop)]
data = data[~data['gt1'].isin(to_drop)]

data = data.drop(labels = ['gt1'], axis=1)
featuredata=pd.read_csv('D:/Research Program/feature_data.csv')

#check the null values 
print(data.isnull().sum())
if(data.isnull().sum().any() !=0):
    print("We have null values")
# making new data frame with dropped NA values
df = data.dropna(axis = 0, how ='any')
  
# comparing sizes of data frames
print("Old data frame length:", len(data), "\nNew data frame length:", 
       len(df), "\nNumber of rows with at least 1 NA value: ",
       (len(data)-len(df)))


N_TIME_STEPS = 128
N_FEATURES = 6
step = 2
segments = []
labels = []
acc_per_fold = []
loss_per_fold = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xa = df['x'].values[i: i + N_TIME_STEPS]
    ya = df['y'].values[i: i + N_TIME_STEPS]
    za = df['z'].values[i: i + N_TIME_STEPS]
    xg = df['x1'].values[i: i + N_TIME_STEPS]
    yg = df['y1'].values[i: i + N_TIME_STEPS]
    zg = df['z1'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['gt'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xa, ya, za, xg, yg, zg])
    labels.append(label)
    
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
# Define the K-fold Cross Validator
loso = LeaveOneOut()

model = keras.Sequential()
model.add(LSTM(150, input_shape=(n_timesteps,n_features)))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))


model.summary()

for train, test in loso.split(reshaped_segments, labels):
    model = keras.Sequential()
    model.add(LSTM(64,input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_CNN = model.fit(reshaped_segments[train],
                    labels[train],
                    epochs=30,
                    batch_size=64,
                    verbose=1)
    # Generate generalization metrics
    scores = model.evaluate(reshaped_segments[test], labels[test], verbose=0)
    print('Score'': {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')