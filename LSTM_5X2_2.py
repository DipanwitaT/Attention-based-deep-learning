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
from sklearn.metrics import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import keras
from keras.wrappers.scikit_learn import KerasClassifier
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
step = 20
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
# Define the K-fold Cross Validator
inner_CV = KFold(n_splits=2, shuffle=True, random_state=42)
outer_CV = KFold(n_splits=5, shuffle=True, random_state=42)
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
# K-fold Cross Validation model evaluation
fold_no = 1
n_outputs=2

def create_model():
    model = keras.Sequential()
    model.add(LSTM(32,input_shape=(128,6)))
    model.add(LSTM(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for train, test in inner_CV.split(reshaped_segments, labels):
    model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 5x2-fold cross validation
 
    results = cross_val_score(model, reshaped_segments[test], labels[test], cv=outer_CV).mean()
    print(results)