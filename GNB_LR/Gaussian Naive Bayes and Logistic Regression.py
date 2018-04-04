# -*- coding: utf-8 -*-
"""

@author: Mandar Rajendra Kulkarni

"""


# Library imports
from __future__ import division

import csv
import numpy as np
import math
from random import randrange
from matplotlib import pyplot


# Function for Evaluates Gaussian Probabilites 
def gaussian_distribution_val(mean, variance, x):
    
    output = 0
    
    if (variance != 0):
        y = np.sqrt(2*math.pi*variance)
        w = math.pow(y, -1)
        z = math.exp(float((-(-mean + float(x))**2)/(2*variance)))
        output = z*w
    return output

# Function for calculating prior probabilities 
def prior_probab(input_list, z):
    
    zeros = 0
    ones = 0
    for i in input_list:
        if (float(i) == 0):
            zeros = zeros + 1
        else:
            ones = ones + 1
    if z==0:
        p = float(zeros/len(input_list))
    else:
        p = float(ones/len(input_list))
    return p
  
    
# Function for calculating mean and variance of input training dataset
def Mean_Variance(input_list):
    
    mean = 0
    variance = 0
    if len(input_list) > 0:
        mean = float(sum(input_list)/len(input_list))
        variance = 0
        for i in input_list:
            variance += (i-mean)**2
        variance = float(variance/(len(input_list)))
    
    return mean, variance


# Function for calculating features with given value of Y label
def feature_pass(input_list, z):
    
    output = []

    for i in range(len(z)):
        if(int(z[i]) == 1):
            output.append(float(input_list[i]))
    return output


# Function for calculating the features with given value of Y label
def feature_fail(input_list, z):
    output = []

    for i in range(len(z)):
        if(int(z[i]) == 0):
            output.append(input_list[i])
    return output

    
# Function for calculating accuracy of Naive Bayes classifier
def accuracy_naive_bayes(test_data_input, naive_results_test):
    
    correct_classifications = 0
    for i in range(len(test_data_input)):
        if int(test_data_input[i][4]) == naive_results_test[i]:
            correct_classifications += 1
    accuracy = float(correct_classifications/len(test_data_input))
    return accuracy


# Function for reading 2/3rd of the 3 fold data
def read_data():
    
    features = []
    train_data = []
    test_data = []
   
    with open('dataset.txt') as csv_file:
    
        csv_read = csv.reader(csv_file, delimiter = ',')

        for line in csv_read:
            features.append(line)
            test_data.append(line)
    
    for i in range(int(915)):        
        random = randrange(len(features))
        train_data.append(features.pop(random))        
        
    return train_data, test_data


# Function for implementation of Gaussian Naive Bayes Classifier
def GNB_classifier_random():
    
    test_data = []
    accuracy = []
    train_data = []  
    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

# Reading data from the file    
    train_data, test_data = read_data()
  
# Implementation for various fractions of data
    for fraction in fractions:
        feature_one = []
        feature_two = []
        feature_three = []
        feature_four = []
        features = []
        naive_results = []
        y = []
        output = []
        
# Taking 5 readings per fraction  
        for j in range(0, 5):
    
            for i in range(int(fraction*len(train_data))):
                
                random = randrange(len(train_data))
                features.append((train_data[random][0:4]))
                y.append((train_data[random][4]))
            
# Finding conditional probabilities and mean and variance for features        
            for i in range(len(y)):
                feature_one.append(float(features[i][0]))
                feature_two.append(float(features[i][1]))
                feature_three.append(float(features[i][2]))
                feature_four.append(float(features[i][3]))

          
            feature_one_pass = feature_pass(feature_one, y)
            feature_one_fail = feature_fail(feature_one, y)
            feature_two_pass = feature_pass(feature_two, y)
            feature_two_fail = feature_fail(feature_two, y)
            feature_three_pass = feature_pass(feature_three, y)
            feature_three_fail = feature_fail(feature_three, y)
            feature_four_pass = feature_pass(feature_four, y)
            feature_four_fail = feature_fail(feature_four, y)
        
            
            mean_var_x1_pass = Mean_Variance(feature_one_pass)
            mean_var_x2_pass = Mean_Variance(feature_two_pass)
            mean_var_x3_pass = Mean_Variance(feature_three_pass)
            mean_var_x4_pass = Mean_Variance(feature_four_pass)
            
            mean_var_x1_fail = Mean_Variance(feature_one_fail)
            mean_var_x2_fail = Mean_Variance(feature_two_fail)
            mean_var_x3_fail = Mean_Variance(feature_three_fail)
            mean_var_x4_fail = Mean_Variance(feature_four_fail)
            	
                      
    
# Calculation of prior probabilities
            prior_fail = prior_probab(y, 0)
            prior_pass = prior_probab(y, 1)
            
            output_zeros = 0
            output_ones = 0

# Calculation of Gaussian Probabilities for each parameter        
            for i in range(len(test_data)):
                x1 = gaussian_distribution_val(mean_var_x1_pass[0], mean_var_x1_pass[1], (test_data[i][0]))
                x2 = gaussian_distribution_val(mean_var_x2_pass[0], mean_var_x2_pass[1], (test_data[i][1]))
                x3 = gaussian_distribution_val(mean_var_x3_pass[0], mean_var_x3_pass[1], (test_data[i][2]))
                x4 = gaussian_distribution_val(mean_var_x4_pass[0], mean_var_x4_pass[1], (test_data[i][3]))
                
                x5 = gaussian_distribution_val(mean_var_x1_fail[0], mean_var_x1_fail[1], (test_data[i][0]))
                x6 = gaussian_distribution_val(mean_var_x2_fail[0], mean_var_x2_fail[1], (test_data[i][1]))
                x7 = gaussian_distribution_val(mean_var_x3_fail[0], mean_var_x3_fail[1], (test_data[i][2]))
                x8 = gaussian_distribution_val(mean_var_x4_fail[0], mean_var_x4_fail[1], (test_data[i][3]))
        
                result_pass = x1*x2*x3*x4*(prior_pass)
                result_fail = x5*x6*x7*x8*(prior_fail)
                
                
                if result_pass > result_fail:
                    output_ones += 1
                    naive_results.append(1)
         
                else:
                    output_zeros += 1
                    naive_results.append(0)
        
        
            output.append(accuracy_naive_bayes(test_data, naive_results))
        
        accuracy.append(float(sum(output)/len(output)))
    print('Accuracy for Naive Bayes Over increasing order of fraction of training data', accuracy) 
    pyplot.plot(fractions, accuracy)

# Generating data based on the mean and variance of the training dataset
        
    gen_x1 = np.random.normal(mean_var_x1_pass[0], mean_var_x1_pass[1], 400) 
    gen_x1_mean_var = Mean_Variance(gen_x1)
    
    gen_x2 = np.random.normal(mean_var_x2_pass[0], mean_var_x2_pass[1], 400) 
    gen_x2_mean_var = Mean_Variance(gen_x2)

    gen_x3 = np.random.normal(mean_var_x3_pass[0], mean_var_x3_pass[1], 400) 
    gen_x3_mean_var = Mean_Variance(gen_x3)

    gen_x4 = np.random.normal(mean_var_x4_pass[0], mean_var_x4_pass[1], 400) 
    gen_x4_mean_var = Mean_Variance(gen_x4)
    
    print('Mean and variance of the generated samples for feature x1, x2, x3 and x4',
          gen_x1_mean_var, gen_x2_mean_var, gen_x3_mean_var, gen_x4_mean_var)  


    print('Mean and variance of the original samples for feature x1, x2, x3 and x4',
      mean_var_x1_pass, mean_var_x2_pass, mean_var_x3_pass, mean_var_x4_pass) 


# Function for predicting the class in Logistic regression
def y_predict(e, coeffecients):
 

    y_predicted = coeffecients[0]
    
    for i in range(0,len(e)-1):
        y_predicted += coeffecients[i+1]*float(e[i])
        y_pred = float(math.exp(y_predicted)/(1+math.exp(y_predicted)))

    return y_pred
  
 
# # Function for training coefficients
    
def coefficient_training(input_list):
    
    learning_rate = 0.0001
    coeffecients = []
    
    for i in range(len(input_list[0])):
        coeffecients.append(0)

# Performing 2000 iterations    
    for j in range(0, 2000):
        
        y_predicted = []
        for e in (input_list):
            #print('E', e[4])
            y_predicted.append(y_predict(e, coeffecients))

        for i in range(0,len(e)-1):
            for j in range(len(input_list)):            
            
                coeffecients[0] += learning_rate*(float(input_list[j][4])-y_predicted[j])
                coeffecients[i+1] += learning_rate*float(input_list[j][i])*(float(input_list[j][4])-y_predicted[j])
            
   
    return coeffecients        

# Logistic regression training

def logistic_regression():
    
    test_data = []    
    train_data = []    
    fractions =  [0.01, 0.02, 0.05, 0.1, 0.625, 1]
    accuracy = []
    
    train_data, test_data = read_data()    
    
# Implementation for different fractions of data
    
    for fraction in fractions:
        output = []
        for m in range(0, 5):
            features = []
            results = []
            for i in range(int(fraction*915)):
            
                random = randrange(len(train_data))
                features.append(train_data[random])

# Obtaining trained coefficients
                          
            coeff = coefficient_training(features)
            
            for i in test_data:
                res = y_predict(i, coeff)
                if res > 0.5:
                    results.append(1)
                else:
                    results.append(0)
                        
            output.append(accuracy_naive_bayes(test_data, results))
      
# Calculating accuracy over 5 iterations
            
        accuracy.append(float(sum(output)/len(output)))
    pyplot.plot(fractions, accuracy)
    print('Logistic Regression Accuracy over increasing order of fractions', accuracy)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Fractions of training set')
    
# Calling the required functions - main()  
    
GNB_classifier_random()
logistic_regression()

