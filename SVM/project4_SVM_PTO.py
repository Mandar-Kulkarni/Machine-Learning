import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics

def data_read():
    # Change the path to the train set X
    data_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_PTO.mat')
    # Change the path to the train set Label
    label_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category PTO.mat')

    data_PTO = data_PTO['PTO']
    label_PTO = label_PTO['Category']
    return data_PTO, label_PTO

def test_read():

    # Change the path to test set
    data_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/dataset_PTO.mat')
    label_PTO = sio.loadmat('D:/E Drive/ASU Study/EEE 511 Artificial Neural Computation/Project_4/Category PTO.mat')

    data_PTO = data_PTO['PTO']
    label_PTO = label_PTO['Category']
    return data_PTO, label_PTO

def accuracy(Y_test, Y_pred):
    acc = metrics.accuracy_score(Y_test, Y_pred)
    F1 = metrics.f1_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.precision_score(Y_test, Y_pred)

    return acc, F1, precision, recall

# Model training

def SVM(X_train, Y_train, X_test, Y_test):
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, Y_train)

    Y_pred = svm_classifier.predict(X_test)
    return Y_pred, svm_classifier


def rearrange_label(Y):
    ret = []
    for i in Y:
        ret.append(i[0])
    return ret

if __name__ == '__main__':
    X, Y = data_read()
    kfolds = KFold(n_splits=10, shuffle=True)

# Kfold validation
    for train_index, validation_index in kfolds.split(X):
        X_train, X_test = X[train_index], X[validation_index]
        Y_train, Y_test = Y[train_index], Y[validation_index]

        X_train = np.array(X_train).tolist()
        Y_train = rearrange_label(Y_train)
        Y_test = rearrange_label(Y_test)

        Y_pred, svm_classifier = SVM(X_train, Y_train, X_test, Y_test)

        acc, F1, precision, recall = accuracy(Y_test, Y_pred)

    # Read the test set and predict the values and print accuracy, F1, precision, recall

    X_test, Y_test = test_read()
    Y_pred = svm_classifier.predict(X_test)
    Y_test = rearrange_label(Y_test)
    acc, F1, precision, recall = accuracy(Y_test, Y_pred)
    print("Accuracy, F1, Precision, Recall of the Test Set", acc, F1, precision, recall)