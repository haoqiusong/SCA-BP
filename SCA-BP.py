# SCA-BP

import pandas as pd
import numpy  as np
import math
import random
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Control the scale of the searching space
min_values = -5
max_values = 5

global dimension
dimension = 0


# Sine Cosine Algorithm
# Function: Initialize Variables
def initial_position(solutions = 10, dim = dimension):
    position = pd.DataFrame(np.zeros((solutions, dim)))
    position['Fitness'] = 0.0
    p=pd.DataFrame(np.array(b))
    for i in range(0, solutions):
        position.iloc[i] = p[0]
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Initialize Destination Position
def destination_position(dimension = dimension):
    destination = pd.DataFrame(np.zeros((1, dimension)))
    destination['Fitness'] = 0.0
    destination.iloc[0,-1] = target_function(destination.iloc[0,0:destination.shape[1]-1])
    return destination

# Function: Update Destination by Fitness
def update_destination(position, destination):
    updated_position = position.copy(deep = True)
    for i in range(0, position.shape[0]):
        if (updated_position.iloc[i,-1] < destination.iloc[0,-1]):
            destination.iloc[0] = updated_position.iloc[i]
    return destination

# Function: Update Position
def update_position(position, destination, r1 = 2, dim = dimension):
    updated_position = position.copy(deep = True)

    for i in range(0, updated_position.shape[0]):
        for j in range (0, dim):

            r2 = 2 * math.pi * (int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1))
            r3 = 2 * (int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1))
            r4 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            
            if (r4 < 0.5):
                updated_position.iloc[i,j] = updated_position.iloc[i,j] + \
                (r1 * math.sin(r2) * abs(r3 * destination.iloc[0,j] - updated_position.iloc[i,j]))
                if (updated_position.iloc[i,j] > max_values):
                    updated_position.iloc[i,j] = max_values
                elif (updated_position.iloc[i,j] < min_values):
                    updated_position.iloc[i,j] = min_values 
            else:
                updated_position.iloc[i,j] = updated_position.iloc[i,j] + \
                (r1 * math.cos(r2) * abs(r3 * destination.iloc[0,j] - updated_position.iloc[i,j]))
                if (updated_position.iloc[i,j] > max_values):
                    updated_position.iloc[i,j] = max_values
                elif (updated_position.iloc[i,j] < min_values):
                    updated_position.iloc[i,j] = min_values        
        
        updated_position.iloc[i,-1] = target_function(updated_position.iloc[i,0:updated_position.shape[1]-1])
            
    return updated_position

# SCA Function
def sine_cosine_algorithm(solutions = 10, a_linear_component = 2,  dim = dimension, iterations = 50):
    count = 1
    position = initial_position(solutions = solutions, dim = dim)
    destination = destination_position(dimension = dim)

    while (count <= iterations):
        print(count)
        r1 = a_linear_component - count * (a_linear_component / iterations)
        destination = update_destination(position, destination)
        position = update_position(position, destination, r1 = r1, dim = dimension)
        count = count + 1 
    
    return destination.iloc[destination['Fitness'].idxmin()]

# Function to be Minimized.
def target_function (variables_values = np.zeros(dimension)):
    variables_values = variables_values.tolist()
    result = target.simul(X_train,y_train,variables_values)
    total = 0
    l = len(result[0][0])
    result[0][0] = result[0][0].tolist()
    for i in range(l):
        total += result[0][0][i]**2
    
    return total


# BP neural network
def logistic(x):
    return np.exp(x) / (1 + np.exp(x))

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))
   
class BP():
    
    def __init__(self,layers,batch):
                
        self.layers = layers
        self.batch = batch
        self.activation = logistic
        self.activation_deriv = logistic_derivative
        
        self.num_layers = len(layers)
        self.biases = [np.random.randn(x) for x in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
        
    def predict(self, x):
        
        for b, w in zip(self.biases, self.weights):
            # Calculate the value, sum of weights plus bias
            z = np.dot(x, w) + b
            # Calculate the output value
            x = self.activation(z)
        return self.classes_[np.argmax(x, axis=1)] # For every sample, calculate which class it belongs to.
        
    def simul(self,X,y,best_position):
        
        # temporary variables
        weights_temp = []
        c = []
        d = []
        temp = 0
        
        # Store and split the weights
        for l,m in zip(self.layers[:-1], self.layers[1:]):
            for i in range(l):
                for j in range(m):
                    c.append(best_position[temp])
                    temp = temp + 1
                d.append(c)
                c = []
            weights_temp.append(np.array(d))
            d = []
        
        labelbin = LabelBinarizer()
        y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        training_data = [(x,y) for x, y in zip(X, y)]
        n = len(training_data)
        batches = [training_data[k:k+self.batch] for k in range(0, n, self.batch)]
        for mini_batch in batches:
            x = []
            y = []
            for a,b in mini_batch:
                x.append(a)
                y.append(b)
            activations = [np.array(x)]
            for b, w in zip(self.biases, weights_temp): 
                z = np.dot(activations[-1],w) + b 
                output = self.activation(z)
                activations.append(output)
            error = activations[-1] - np.array(y)
            deltas = [error * self.activation_deriv(activations[-1])] # error rate of the output layer
        return deltas

    def fit(self, X, y, learning_rate=0.1, epochs=10):
    
        # Change the self.biases to another array for future training
        global b
        b = []
        for i in self.biases:
            arr = i.tolist()
            b.append(arr)
            
        # Change the format of the global variable b
        b_temp = str(b)
        b_temp = b_temp.replace('[','')
        b_temp = b_temp.replace(']','')
        b = list(eval(b_temp))
        
        # Calculate the dimension
        co = 1 # temporary variable
        dimension = 0
        for i in zip(self.layers[:-1], self.layers[1:]):
            for j in i:
                co = co * j
            dimension = dimension + co
            co = 1
        sca = sine_cosine_algorithm(solutions = 10, a_linear_component = 2,  dim = dimension, iterations = 50)
        
        aa = []
        c = []
        d = []
        temp = 0
        for l,m in zip(self.layers[:-1], self.layers[1:]):
            for i in range(l):
                for j in range(m):
                    c.append(sca[temp])
                    temp = temp + 1
                d.append(c)
                c = []
            aa.append(np.array(d))
            d = []
        self.weights = aa
        
        labelbin = LabelBinarizer()
        y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        training_data = [(x,y) for x, y in zip(X, y)]
        n = len(training_data)
        for k in range(epochs):
            # Shuffling the training sets
            random.shuffle(training_data)
            batches = [training_data[k:k+self.batch] for k in range(0, n, self.batch)]
            # Batch Gradient descent
            for mini_batch in batches:
                x = []
                y = []
                for a,b in mini_batch:
                    x.append(a)
                    y.append(b)
                activations = [np.array(x)]
                # Back-propogation
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(activations[-1],w) + b 
                    # Calculate the output value
                    output = self.activation(z)
                    # Put the output into the input list for updating the weights
                    activations.append(output)
                # Calculate the error
                error = activations[-1] - np.array(y)
                # Calculate the error rate of the output layer
                deltas = [error * self.activation_deriv(activations[-1])]
                
                # Calculate the error rates of the hidden layer
                for l in range(self.num_layers-2, 0, -1):
                    deltas.append(self.activation_deriv(activations[l]) * np.dot(deltas[-1],self.weights[l].T))
                deltas.reverse()
                
                # Update the weights and biases
                for j in range(self.num_layers-1):
                    # delta weights
                    delta = learning_rate / self.batch * ((np.atleast_2d(activations[j].sum(axis=0)).T).dot(np.atleast_2d(deltas[j].sum(axis=0))))
                    # Update the weights
                    self.weights[j] -= delta
                    # delta biases = learning rate * error rate
                    delta = learning_rate / self.batch * deltas[j].sum(axis=0)
                    # Update the biases
                    self.biases[j] -= delta
        return self   


# Data preprocessing
X = [] # samples
Y = [] # labels

for i in range(0, 6):
    for f in os.listdir("/Users/songhaoqiu/Desktop/p/%s" % i):
        # Open an image
        Images = cv2.imread("/Users/songhaoqiu/Desktop/p/%s/%s" % (i, f)) 
        image=cv2.resize(Images,(256,256))
        # Calculate different histograms from color blue and green
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0])
        X.append((hist/255).flatten())
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
# Split the training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# Prediction
target = BP([X_train.shape[1],6], 10)
clf = target.fit(X_train, y_train, epochs = 100)
predictions_labels = clf.predict(X_test)
print(confusion_matrix(y_test, predictions_labels))
print(classification_report(y_test, predictions_labels))