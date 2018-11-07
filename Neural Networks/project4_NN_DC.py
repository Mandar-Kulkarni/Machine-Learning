# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.neural_network import MLPClassifier

def data_read():
    # Change the input path of the train data set X
    data_DC = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_DC.mat')
    # Change the path to the train set label
    label_DC = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category DC.mat')

    data_DC = data_DC['DC']
    label_DC = label_DC['Category']
    return data_DC, label_DC

def test_read():
    # Change the input path of the test data set
    data_DC = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_DC.mat')
    label_DC = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category DC.mat')

    data_DC = data_DC['DC']
    label_DC = label_DC['Category']
    return data_DC, label_DC

def rearrange_label(Y):
    ret = []
    for i in Y:
        ret.append(i[0])
    return ret


# Training and Kfold validation
def keras_NN(X, Y):

    kfolds = KFold(n_splits=10, shuffle=True)

    accuracy = []

    model = keras.Sequential()
    # 2 hidden layers
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    for train_index, validation_index in kfolds.split(X):
        X_train, X_test = X[train_index], X[validation_index]
        Y_train, Y_test = Y[train_index], Y[validation_index]

        model.fit(X_train, Y_train, epochs=50, batch_size=10)

        scores = model.evaluate(X_test, Y_test)
        accuracy.append(scores[1])
        #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


    print(accuracy)
    return model


def rearrange_label(Y):
    ret = []
    for i in Y:
        ret.append(i[0])
    return ret

if __name__ == '__main__':
    X, Y = data_read()


# Read test set and test
    model = keras_NN(X, Y)

    X_test, Y_test = test_read()

    scores = model.evaluate(X_test, Y_test)
    print("Accuracy of the test set")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
