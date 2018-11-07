import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics

def data_read():
    # Change the path to the train set X
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


def accuracy(Y_test, Y_pred):
    acc = metrics.accuracy_score(Y_test, Y_pred)
    F1 = metrics.f1_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred)
    recall = metrics.precision_score(Y_test, Y_pred)

    return acc, F1, precision, recall

# Training the non-linear model
def SVM(X_train, Y_train, X_test, Y_test):
    svm_classifier = svm.NuSVC()
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

    for train_index, validation_index in kfolds.split(X):
        X_train, X_test = X[train_index], X[validation_index]
        Y_train, Y_test = Y[train_index], Y[validation_index]

        X_train = np.array(X_train).tolist()
        Y_train = rearrange_label(Y_train)
        Y_test = rearrange_label(Y_test)

        Y_pred, svm_classifier = SVM(X_train, Y_train, X_test, Y_test)

        acc, F1, precision, recall = accuracy(Y_test, Y_pred)


# Reads test set and prints accuracy, F1, recall, precision

    X_test, Y_test = test_read()
    Y_pred = svm_classifier.predict(X_test)
    Y_test = rearrange_label(Y_test)
    acc, F1, precision, recall = accuracy(Y_test, Y_pred)
    print("Accuracy, F1, Precision, Recall of the Test Set", acc, F1, precision, recall)
