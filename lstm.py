import pandas as pd
import sklearn.metrics as metrique
from pandas import Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import LSTM, Dense, Embedding, Dropout,Input, Layer, Concatenate, Permute, Dot, Multiply, Flatten,ReLU
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import Sequential
from keras import backend as K, regularizers, Model, metrics
from keras.backend import cast

data = pd.read_csv('newdatset.csv', na_filter=True)
x = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.2, random_state=2)
model = Sequential()
model.add(Dense(100, input_shape=(30,)))
model.add(ReLU())
model.add(Dense(100))
model.add(ReLU())
model.add(Dense(10))
model.add(ReLU())
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='Adam', loss='categorical_hinge', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=15)
y_predicted = (model.predict(X_test) >= 0.5)
conf_mat = confusion_matrix(Y_test, y_predicted)



from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
print(conf_mat)

for i in y_predicted:
    print(i)



acc = accuracy_score(Y_test, y_predicted)
f1score = f1_score(Y_test, y_predicted)
recall = recall_score(Y_test, y_predicted)
precisionscore = precision_score(Y_test, y_predicted)

print(acc,f1score,recall,precisionscore)