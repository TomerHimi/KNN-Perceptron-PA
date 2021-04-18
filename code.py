"""
Created on Fri Nov 20 13:13:27 2020
@author: Tomer Himi

"""
import numpy as np
import random
import sys
random.seed(0)

wine_type = {'R': 0, 'W': 1}
k = 7
perceptron_weights = [np.zeros((1, 13)), np.zeros((1, 13)), np.zeros((1, 13))]
perceptron_eta = 0.1
pa_weights = [np.zeros((1, 13)), np.zeros((1, 13)), np.zeros((1, 13))]

def formatted_data(data):
    """pre-proccesing function of raw data
    parm: data: array of training or testing data
    type: new_centroids: ndarray
    return: formmated array for learning
    rtype: data: ndarray"""
    #change categorial coulmn to numeric one
    data[:,-1] = [wine_type[item] for item in data[:,-1]]   
    data = data.astype(np.float)
    #min-max normaliztion
    for feature in range(data.shape[1]):
        column = data[:, feature]
        min_column = np.min(column)
        max_column = np.max(column)
        for index, item in enumerate(column):
            column[index] = (item - min_column) / (max_column - min_column)
    #add ones column for the bias
    data = np.append(data, np.ones([data.shape[0], 1]), axis = 1)
    return data

def train_knn(training_data, label_data, test_data):
    """implementation function of KNN algorithm
    parm: training_data: array of training data
    parm: label_data: array of labels 
    parm: test_data: array of testing data
    type: training_data: ndarray
    type: label_data: ndarray
    type: test_data: ndarray
    return: array of labels for the testing data
    rtype: ndarray"""        
    predict_labels = []  
    for x_test in test_data:
        distances = np.sqrt(np.sum((training_data - x_test) ** 2, axis = 1))
        min_index = np.argmin(distances)
        predict_labels.append(label_data[min_index])
    return predict_labels

def train_perceptron(training_data, label_data):
    """implementation function of Perceptron algorithm
    parm: training_data: array of training data
    parm: label_data: array of labels 
    type: training_data: ndarray
    type: label_data: ndarray"""
    data_list = list(zip(training_data, label_data))
    for x in range(100):
         random.shuffle(data_list)
         for i, data in enumerate(data_list):
            class_index = np.argmax(np.dot(perceptron_weights, data[0]))
            if data[1] != class_index:
                perceptron_weights[int(data[1])] = perceptron_weights[int(data[1])] + (perceptron_eta * data[0])
                perceptron_weights[class_index] = perceptron_weights[class_index] - (perceptron_eta * data[0])

def test_perceptron(test_data):
    """testing function for new examples after learning by Perceptron 
    parm: test_data: array of testing data
    type: test_data: ndarray
    return: array of labels for the testing data
    rtype: ndarray"""
    labels = np.zeros((1, len(test_data)))
    for i, data in enumerate(test_data):
        y_hat = np.argmax(np.dot(perceptron_weights, data))
        labels[0,i] = y_hat
    return labels

def train_pa(training_data, label_data):
    """implementation function of Passive Aggresive algorithm
    parm: training_data: array of training data
    parm: label_data: array of labels 
    type: training_data: ndarray
    type: label_data: ndarray"""
    data_list = list(zip(training_data, label_data))
    loss = 0
    random.shuffle(data_list)
    for i, data in enumerate(data_list):
        class_index = np.argmax(np.dot(pa_weights, data[0]))
        new_data = int(data[1])
        w_y = np.dot(pa_weights[new_data], np.transpose(data[0]))
        w_y_hat = np.dot(pa_weights[class_index], np.transpose(data[0]))
        arr = w_y_hat - w_y
        num = 1 + arr[0]
        if num > 0:
            loss = num
        x_norm = np.linalg.norm(data[0])
        tau = loss / (2 * np.power(x_norm,2))
        if data[1] != class_index:
            pa_weights[int(data[1])] = pa_weights[int(data[1])] + (tau * data[0])
            pa_weights[class_index] = pa_weights[class_index] - (tau * data[0])

def test_pa(test_data):
    """testing function for new examples after learning by Passive Aggresive 
    parm: test_data: array of testing data
    type: test_data: ndarray
    return: array of labels for the testing data
    rtype: ndarray"""
    labels = np.zeros((1, len(test_data)))
    for i, data in enumerate(test_data):
        y_hat = np.argmax(np.dot(pa_weights, data))
        labels[0,i] = y_hat
    return labels

def main():
    raw_data = np.genfromtxt(sys.argv[1], names = None, delimiter = ',',dtype = str)
    training = formatted_data(raw_data)
    labels = np.genfromtxt(sys.argv[2], names = None, delimiter = ',',dtype = int)
    raw_test = np.genfromtxt(sys.argv[3], names = None, delimiter = ',',dtype = str)
    testing = formatted_data(raw_test)
    #training step for each one of the algorithms
    knn_vec = train_knn(training, labels[:len(training)], testing)
    train_perceptron(training, labels[:len(training)])
    train_pa(training, labels[:len(training)])
    #testing step for each one of the algorithms
    perceptron_vec = test_perceptron(testing)
    pa_vec = test_pa(testing)
    #printing labels of testing data for each one of the algorithms
    n = perceptron_vec.size
    for i in range(n):
        print(f"knn: {knn_vec[i]}, perceptron: {int(perceptron_vec[0,i])}, pa: {int(pa_vec[0,i])}")

if __name__ == "__main__":
    main()