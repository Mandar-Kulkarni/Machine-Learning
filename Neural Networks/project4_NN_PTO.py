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
    # Change the path to the train set X
    data_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_PTO.mat')
    # Change the path to the train set label
    label_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category PTO.mat')

    data_PTO = data_PTO['PTO']
    label_PTO = label_PTO['Category']
    return data_PTO, label_PTO

def test_read():
    # Change the path to the test set
    data_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_PTO.mat')
    label_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category PTO.mat')

    data_PTO = data_PTO['PTO']
    label_PTO = label_PTO['Category']
    return data_PTO, label_PTO

def rearrange_label(Y):
    ret = []
    for i in Y:
        ret.append(i[0])
    return ret


def accuracy(Y_test, Y_pred):
    acc = metrics.accuracy_score(Y_test, Y_pred)
    F1 = metrics.f1_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.precision_score(Y_test, Y_pred)

    return acc, F1, precision, recall


def rearrange_label(Y):
    ret = []
    for i in Y:
        ret.append(i[0])
    return ret

# Training the model and Kfold validation
def sklearn_NN(X, Y):
    kfolds = KFold(n_splits=10, shuffle=True)


    # Fit model
    classfier = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(64,2), random_state=1)
    for train_index, validation_index in kfolds.split(X):
        X_train, X_test = X[train_index], X[validation_index]
        Y_train, Y_test = Y[train_index], Y[validation_index]

        X_train = np.array(X_train).tolist()
        Y_train = rearrange_label(Y_train)
        Y_test = rearrange_label(Y_test)

        classfier.fit(X_train, Y_train)
        Y_pred = classfier.predict(X_test)

        Y_pred = np.array(Y_pred).tolist()
        acc = metrics.accuracy_score(Y_test, Y_pred)

        F1 = metrics.f1_score(Y_test, Y_pred)
        precision = metrics.precision_score(Y_test, Y_pred)
        recall = metrics.precision_score(Y_test, Y_pred)
        #print("Accuracy, F1, Precision, Recall", acc, F1, precision, recall)
    return classfier


if __name__ == '__main__':
    X, Y = data_read()

    classifier = sklearn_NN(X, Y)

# Read the test set and predict the values and print accuracy, F1, precision, recall

    X_test, Y_test = test_read()

    Y_test = rearrange_label(Y_test)
    Y_pred = classifier.predict(X_test)

    Y_pred = np.array(Y_pred).tolist()
    acc = metrics.accuracy_score(Y_test, Y_pred)


    F1 = metrics.f1_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.precision_score(Y_test, Y_pred)
    print("Accuracy, F1, Precision, Recall of Test Set", acc, F1, precision, recall)


