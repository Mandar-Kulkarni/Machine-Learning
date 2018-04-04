# -*- coding: utf-8 -*-
'''
Name - Mandar Rajendra Kulkarni

'''
import csv
import numpy as np
from matplotlib import pyplot as plt


# Data pre-processing on the MNIST data set

def get_data():
   
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    with open('mnist_train.csv', 'rb') as train_file:
        csv_read = csv.reader(train_file)
        for i,line in enumerate(csv_read):
            if i < 6000:
                train_data.append(line[1:])
                train_label.append(line[0])
            else:
                break
        
    with open('mnist_test.csv', 'rb') as test_file:
        csv_reader = csv.reader(test_file)
        for i,line in enumerate(csv_reader):
            if i < 1000:
                test_data.append(line[1:])
                test_label.append(line[0])
            else:
                break
    return train_data, train_label, test_data, test_label

# Calculating Accuracy

def get_accuracy(test_label, closest_label):
    
    correct_classification = 0
    for i in range(len(test_data)):
        if test_label[i] == closest_label[i]:
            correct_classification += 1
    return float(correct_classification)/len(closest_label)
    
# Estimate the label of the test data

def find_estimate(train_data, train_label, test_data, test_label):
    final_distance = []
    train_data_np = np.array(train_data, dtype = 'int')
    test_data_np = np.array(test_data, dtype = 'int')
    for j in range(len(test_data)):       
        dist = []
        for i in range(len(train_data)):
            dist.append(np.linalg.norm(test_data_np[j]-train_data_np[i]))
        final_distance.append(dist)
    return (np.array(final_distance))
        
# Find the k-nearest values

def k_nearest(train_label, distance_array, knn):
    nearest_neighbor = []
    for i in range(len(distance_array)):
        closest_label = []    
        closest_neighbor = (np.argpartition(distance_array[i], knn)[:knn])   
        for k in closest_neighbor:
            closest_label.append(train_label[k])  

        nearest_neighbor.append(max(closest_label, key = closest_label.count))  
    return get_accuracy(test_label, nearest_neighbor)
 

# Execute for the following values of k

accuracy = []
knns = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
train_data, train_label, test_data, test_label = get_data()
distance_array = find_estimate(train_data, train_label, test_data, test_label)
for f in knns:
    accuracy.append(1-(k_nearest(train_label, distance_array, f)))
print(accuracy)

# Plot accuracies for different values of k

plt.plot(knns, accuracy)
plt.title("Test Error Plot")
plt.xlabel('K')
plt.ylabel('Test Error')


    