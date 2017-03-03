
# 2 Layer Neural Network Regression

In this project created by Yida Wang, 2 layer neural network is used for predicting funds index with the help of SGD training of multiple csv files. It's simple and clean enough for CPU based equipment.


```python
"""
Copyright (c) 2017, Yida Wang
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
"""

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load and prepare the data

A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, I write the code to load and prepare the data.


```python
# Here is where we get the raw data

data_list = ['/Volumes/机器学习/发送文件/发送文件/1.csv', 
             '/Volumes/机器学习/发送文件/发送文件/2.csv', 
             '/Volumes/机器学习/发送文件/发送文件/3.csv', 
             '/Volumes/机器学习/发送文件/发送文件/4.csv', 
             '/Volumes/机器学习/发送文件/发送文件/5.csv', 
             '/Volumes/机器学习/发送文件/发送文件/6.csv', 
             '/Volumes/机器学习/发送文件/发送文件/7.csv', 
             '/Volumes/机器学习/发送文件/发送文件/8.csv', 
             '/Volumes/机器学习/发送文件/发送文件/9.csv', 
             '/Volumes/机器学习/发送文件/发送文件/10.csv',
             '/Volumes/机器学习/发送文件/发送文件/11.csv', 
             '/Volumes/机器学习/发送文件/发送文件/12.csv', 
             '/Volumes/机器学习/发送文件/发送文件/13.csv', 
             '/Volumes/机器学习/发送文件/发送文件/14.csv', 
             '/Volumes/机器学习/发送文件/发送文件/15.csv', 
             '/Volumes/机器学习/发送文件/发送文件/16.csv', 
             '/Volumes/机器学习/发送文件/发送文件/17.csv', 
             '/Volumes/机器学习/发送文件/发送文件/18.csv', 
             '/Volumes/机器学习/发送文件/发送文件/19.csv',
             '/Volumes/机器学习/发送文件/发送文件/20.csv',
             '/Volumes/机器学习/发送文件/发送文件/21.csv', 
             '/Volumes/机器学习/发送文件/发送文件/22.csv', 
             '/Volumes/机器学习/发送文件/发送文件/23.csv', 
             '/Volumes/机器学习/发送文件/发送文件/24.csv', 
             '/Volumes/机器学习/发送文件/发送文件/25.csv', 
             '/Volumes/机器学习/发送文件/发送文件/26.csv', 
             '/Volumes/机器学习/发送文件/发送文件/27.csv', 
             '/Volumes/机器学习/发送文件/发送文件/28.csv', 
             '/Volumes/机器学习/发送文件/发送文件/29.csv', 
             '/Volumes/机器学习/发送文件/发送文件/30.csv',
             '/Volumes/机器学习/发送文件/发送文件/31.csv', 
             '/Volumes/机器学习/发送文件/发送文件/32.csv', 
             '/Volumes/机器学习/发送文件/发送文件/33.csv', 
             '/Volumes/机器学习/发送文件/发送文件/34.csv', 
             '/Volumes/机器学习/发送文件/发送文件/35.csv', 
             '/Volumes/机器学习/发送文件/发送文件/36.csv', 
             '/Volumes/机器学习/发送文件/发送文件/37.csv', 
             '/Volumes/机器学习/发送文件/发送文件/38.csv', 
             '/Volumes/机器学习/发送文件/发送文件/39.csv', 
             '/Volumes/机器学习/发送文件/发送文件/40.csv',
             '/Volumes/机器学习/发送文件/发送文件/41.csv', 
             '/Volumes/机器学习/发送文件/发送文件/42.csv', 
             '/Volumes/机器学习/发送文件/发送文件/43.csv', 
             '/Volumes/机器学习/发送文件/发送文件/44.csv', 
             '/Volumes/机器学习/发送文件/发送文件/45.csv', 
             '/Volumes/机器学习/发送文件/发送文件/46.csv', 
             '/Volumes/机器学习/发送文件/发送文件/47.csv', 
             '/Volumes/机器学习/发送文件/发送文件/48.csv', 
             '/Volumes/机器学习/发送文件/发送文件/49.csv', 
             '/Volumes/机器学习/发送文件/发送文件/50.csv',
             '/Volumes/机器学习/发送文件/发送文件/51.csv', 
             '/Volumes/机器学习/发送文件/发送文件/52.csv', 
             '/Volumes/机器学习/发送文件/发送文件/53.csv', 
             '/Volumes/机器学习/发送文件/发送文件/54.csv', 
             '/Volumes/机器学习/发送文件/发送文件/55.csv', 
             '/Volumes/机器学习/发送文件/发送文件/56.csv', 
             '/Volumes/机器学习/发送文件/发送文件/57.csv', 
             '/Volumes/机器学习/发送文件/发送文件/58.csv', 
             '/Volumes/机器学习/发送文件/发送文件/59.csv', 
             '/Volumes/机器学习/发送文件/发送文件/60.csv',
             '/Volumes/机器学习/发送文件/发送文件/61.csv', 
             '/Volumes/机器学习/发送文件/发送文件/62.csv', 
             '/Volumes/机器学习/发送文件/发送文件/63.csv', 
             '/Volumes/机器学习/发送文件/发送文件/64.csv', 
             '/Volumes/机器学习/发送文件/发送文件/65.csv', 
             '/Volumes/机器学习/发送文件/发送文件/66.csv', 
             '/Volumes/机器学习/发送文件/发送文件/67.csv', 
             '/Volumes/机器学习/发送文件/发送文件/68.csv', 
             '/Volumes/机器学习/发送文件/发送文件/69.csv',
             '/Volumes/机器学习/发送文件/发送文件/70.csv',
             '/Volumes/机器学习/发送文件/发送文件/71.csv', 
             '/Volumes/机器学习/发送文件/发送文件/72.csv', 
             '/Volumes/机器学习/发送文件/发送文件/73.csv', 
             '/Volumes/机器学习/发送文件/发送文件/74.csv', 
             '/Volumes/机器学习/发送文件/发送文件/75.csv', 
             '/Volumes/机器学习/发送文件/发送文件/76.csv', 
             '/Volumes/机器学习/发送文件/发送文件/77.csv', 
             '/Volumes/机器学习/发送文件/发送文件/78.csv', 
             '/Volumes/机器学习/发送文件/发送文件/79.csv', 
             '/Volumes/机器学习/发送文件/发送文件/80.csv',
             '/Volumes/机器学习/发送文件/发送文件/81.csv', 
             '/Volumes/机器学习/发送文件/发送文件/82.csv', 
             '/Volumes/机器学习/发送文件/发送文件/83.csv', 
             '/Volumes/机器学习/发送文件/发送文件/84.csv', 
             '/Volumes/机器学习/发送文件/发送文件/85.csv', 
             '/Volumes/机器学习/发送文件/发送文件/86.csv', 
             '/Volumes/机器学习/发送文件/发送文件/87.csv', 
             '/Volumes/机器学习/发送文件/发送文件/88.csv', 
             '/Volumes/机器学习/发送文件/发送文件/89.csv', 
             '/Volumes/机器学习/发送文件/发送文件/90.csv',
             '/Volumes/机器学习/发送文件/发送文件/91.csv', 
             '/Volumes/机器学习/发送文件/发送文件/92.csv', 
             '/Volumes/机器学习/发送文件/发送文件/93.csv', 
             '/Volumes/机器学习/发送文件/发送文件/94.csv', 
             '/Volumes/机器学习/发送文件/发送文件/95.csv', 
             '/Volumes/机器学习/发送文件/发送文件/96.csv', 
             '/Volumes/机器学习/发送文件/发送文件/97.csv', 
             '/Volumes/机器学习/发送文件/发送文件/98.csv', 
             '/Volumes/机器学习/发送文件/发送文件/99.csv', 
             '/Volumes/机器学习/发送文件/发送文件/100.csv']

# Here we set 'header' equals to None to indicate that there are no headers in the csv
# file, Here the index is int64 rather than string.
for idx, data_path in enumerate(data_list):
    if idx == 0:
        raw_data = pd.read_csv(data_path, delimiter=',', header=None)
        print('Here we start to load all the data as listed below:')
        print('File %d:' %(idx+1), data_path)
    else:
        raw_data_temp = pd.read_csv(data_path, delimiter=',', header=None)
        raw_data = raw_data.append(raw_data_temp)
        print('File %d:' %(idx+1), data_path)

# Get the target labels in the second column and use remaining columns as feature
targets = raw_data[2]

# the first column is really large which doesn't mean anything and the first second
# column is all the same which makes it impossible to get scaled
features = raw_data.drop([0, 1, 2], axis=1)
```

    Here we start to load all the data as listed below:
    File 1: /Volumes/机器学习/发送文件/发送文件/1.csv


    /usr/local/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


    File 2: /Volumes/机器学习/发送文件/发送文件/2.csv
    File 3: /Volumes/机器学习/发送文件/发送文件/3.csv
    File 4: /Volumes/机器学习/发送文件/发送文件/4.csv
    File 5: /Volumes/机器学习/发送文件/发送文件/5.csv
    File 6: /Volumes/机器学习/发送文件/发送文件/6.csv
    File 7: /Volumes/机器学习/发送文件/发送文件/7.csv
    File 8: /Volumes/机器学习/发送文件/发送文件/8.csv
    File 9: /Volumes/机器学习/发送文件/发送文件/9.csv
    File 10: /Volumes/机器学习/发送文件/发送文件/10.csv
    File 11: /Volumes/机器学习/发送文件/发送文件/11.csv
    File 12: /Volumes/机器学习/发送文件/发送文件/12.csv
    File 13: /Volumes/机器学习/发送文件/发送文件/13.csv
    File 14: /Volumes/机器学习/发送文件/发送文件/14.csv
    File 15: /Volumes/机器学习/发送文件/发送文件/15.csv
    File 16: /Volumes/机器学习/发送文件/发送文件/16.csv
    File 17: /Volumes/机器学习/发送文件/发送文件/17.csv
    File 18: /Volumes/机器学习/发送文件/发送文件/18.csv
    File 19: /Volumes/机器学习/发送文件/发送文件/19.csv
    File 20: /Volumes/机器学习/发送文件/发送文件/20.csv
    File 21: /Volumes/机器学习/发送文件/发送文件/21.csv
    File 22: /Volumes/机器学习/发送文件/发送文件/22.csv
    File 23: /Volumes/机器学习/发送文件/发送文件/23.csv
    File 24: /Volumes/机器学习/发送文件/发送文件/24.csv
    File 25: /Volumes/机器学习/发送文件/发送文件/25.csv
    File 26: /Volumes/机器学习/发送文件/发送文件/26.csv
    File 27: /Volumes/机器学习/发送文件/发送文件/27.csv
    File 28: /Volumes/机器学习/发送文件/发送文件/28.csv
    File 29: /Volumes/机器学习/发送文件/发送文件/29.csv
    File 30: /Volumes/机器学习/发送文件/发送文件/30.csv
    File 31: /Volumes/机器学习/发送文件/发送文件/31.csv
    File 32: /Volumes/机器学习/发送文件/发送文件/32.csv
    File 33: /Volumes/机器学习/发送文件/发送文件/33.csv
    File 34: /Volumes/机器学习/发送文件/发送文件/34.csv
    File 35: /Volumes/机器学习/发送文件/发送文件/35.csv
    File 36: /Volumes/机器学习/发送文件/发送文件/36.csv
    File 37: /Volumes/机器学习/发送文件/发送文件/37.csv
    File 38: /Volumes/机器学习/发送文件/发送文件/38.csv
    File 39: /Volumes/机器学习/发送文件/发送文件/39.csv
    File 40: /Volumes/机器学习/发送文件/发送文件/40.csv
    File 41: /Volumes/机器学习/发送文件/发送文件/41.csv
    File 42: /Volumes/机器学习/发送文件/发送文件/42.csv
    File 43: /Volumes/机器学习/发送文件/发送文件/43.csv
    File 44: /Volumes/机器学习/发送文件/发送文件/44.csv
    File 45: /Volumes/机器学习/发送文件/发送文件/45.csv
    File 46: /Volumes/机器学习/发送文件/发送文件/46.csv
    File 47: /Volumes/机器学习/发送文件/发送文件/47.csv
    File 48: /Volumes/机器学习/发送文件/发送文件/48.csv
    File 49: /Volumes/机器学习/发送文件/发送文件/49.csv
    File 50: /Volumes/机器学习/发送文件/发送文件/50.csv
    File 51: /Volumes/机器学习/发送文件/发送文件/51.csv
    File 52: /Volumes/机器学习/发送文件/发送文件/52.csv
    File 53: /Volumes/机器学习/发送文件/发送文件/53.csv
    File 54: /Volumes/机器学习/发送文件/发送文件/54.csv
    File 55: /Volumes/机器学习/发送文件/发送文件/55.csv
    File 56: /Volumes/机器学习/发送文件/发送文件/56.csv
    File 57: /Volumes/机器学习/发送文件/发送文件/57.csv
    File 58: /Volumes/机器学习/发送文件/发送文件/58.csv
    File 59: /Volumes/机器学习/发送文件/发送文件/59.csv
    File 60: /Volumes/机器学习/发送文件/发送文件/60.csv
    File 61: /Volumes/机器学习/发送文件/发送文件/61.csv
    File 62: /Volumes/机器学习/发送文件/发送文件/62.csv
    File 63: /Volumes/机器学习/发送文件/发送文件/63.csv
    File 64: /Volumes/机器学习/发送文件/发送文件/64.csv
    File 65: /Volumes/机器学习/发送文件/发送文件/65.csv
    File 66: /Volumes/机器学习/发送文件/发送文件/66.csv
    File 67: /Volumes/机器学习/发送文件/发送文件/67.csv
    File 68: /Volumes/机器学习/发送文件/发送文件/68.csv
    File 69: /Volumes/机器学习/发送文件/发送文件/69.csv
    File 70: /Volumes/机器学习/发送文件/发送文件/70.csv
    File 71: /Volumes/机器学习/发送文件/发送文件/71.csv
    File 72: /Volumes/机器学习/发送文件/发送文件/72.csv
    File 73: /Volumes/机器学习/发送文件/发送文件/73.csv
    File 74: /Volumes/机器学习/发送文件/发送文件/74.csv
    File 75: /Volumes/机器学习/发送文件/发送文件/75.csv
    File 76: /Volumes/机器学习/发送文件/发送文件/76.csv
    File 77: /Volumes/机器学习/发送文件/发送文件/77.csv
    File 78: /Volumes/机器学习/发送文件/发送文件/78.csv
    File 79: /Volumes/机器学习/发送文件/发送文件/79.csv
    File 80: /Volumes/机器学习/发送文件/发送文件/80.csv
    File 81: /Volumes/机器学习/发送文件/发送文件/81.csv
    File 82: /Volumes/机器学习/发送文件/发送文件/82.csv
    File 83: /Volumes/机器学习/发送文件/发送文件/83.csv
    File 84: /Volumes/机器学习/发送文件/发送文件/84.csv
    File 85: /Volumes/机器学习/发送文件/发送文件/85.csv
    File 86: /Volumes/机器学习/发送文件/发送文件/86.csv
    File 87: /Volumes/机器学习/发送文件/发送文件/87.csv
    File 88: /Volumes/机器学习/发送文件/发送文件/88.csv
    File 89: /Volumes/机器学习/发送文件/发送文件/89.csv
    File 90: /Volumes/机器学习/发送文件/发送文件/90.csv
    File 91: /Volumes/机器学习/发送文件/发送文件/91.csv
    File 92: /Volumes/机器学习/发送文件/发送文件/92.csv
    File 93: /Volumes/机器学习/发送文件/发送文件/93.csv



```python
# Head of the features
features.head()
```

## Scaling target variables (optional)
To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

The scaling factors are saved so we can go backwards when we use the network for predictions.


```python
quant_features = features.T.index.tolist()
# Store scalings in a dictionary so we can convert back later
mean, std = targets.mean(), targets.std()
scales_target = [mean, std]
targets = (targets - mean)/std

scales_feature = {}
for each in quant_features:
    mean, std = features[each].mean(), features[each].std()
    scales_feature[each] = [mean, std]
    features.loc[:, each] = (features[each] - mean)/std
```

## Projecting data on a compact dimension (optional)
To deal with a high dimensional data, we need to porject a high dimentional data onto a low dimention, here I use PCA as the method.


```python
# singular value decomposition factorises your data matrix such that:
# 
#   features = U*S*V.T     (where '*' is matrix multiplication)
# 
# * U and V are the singular matrices, containing orthogonal vectors of
#   unit length in their rows and columns respectively.
#
# * S is a diagonal matrix containing the singular values of features - these 
#   values squared divided by the number of observations will give the 
#   variance explained by each PC.
#
# * if features is considered to be an (observations, features) matrix, the PCs
#   themselves would correspond to the rows of S^(1/2)*V.T. if features is 
#   (features, observations) then the PCs would be the columns of
#   U*S^(1/2).
#
# * since U and V both contain orthonormal vectors, U*V.T is equivalent 
#   to a whitened version of features.
trace = np.linalg.matrix_rank(features.values)
print('Not a Number is located there: ', np.where(np.isnan(features.values) == True))

U, s, Vt = np.linalg.svd(features.values, full_matrices=False)
V = Vt.T

# PCs are already sorted by descending order 
# of the singular values (i.e. by the
# proportion of total variance they explain)

# Method 1 reconstruction: 
# if we use all of the PCs we can reconstruct the original signal perfectly.
S = np.diag(s)
Mhat = np.dot(U, np.dot(S, V.T))
print('Using all PCs, MSE = %.6G' %(np.mean((features.values - Mhat)**2)))

dim_remain = 700
# Method 2 weak reconstruction: 
# if we use only the first few PCs the reconstruction is less accurate，
# the dimention is remained the same sa before, but some information is
# lost in this reconstruction process.
Mhat2 = np.dot(U[:, :dim_remain], np.dot(S[:dim_remain, :dim_remain], V[:,:dim_remain].T))
print('Not a Number is located there: ', np.where(np.isnan(features.values) == True))
print('Using first few PCs, MSE = %.6G' %(np.mean((features.values - Mhat2)**2)))

# Method 3 dimention reduction: 
# if we use only the first few PCs the reconstruction is less accurate,
# the dimension is also recuded to (or to say projected on) into another 
# low dimenional space.
Mhat3 = np.dot(U[:, :dim_remain], S[:dim_remain, :dim_remain])

features = pd.DataFrame(Mhat3)
targets.index = features.index

```

    Not a Number is located there:  (array([], dtype=int64), array([], dtype=int64))
    Using all PCs, MSE = 4.10792E-29
    Not a Number is located there:  (array([], dtype=int64), array([], dtype=int64))
    Using first few PCs, MSE = 2.08115E-15


## Splitting the data into training, testing, and validation sets

We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set). We'll save the last 400 samples of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.


```python
# Save the last 400 as test data
test_features, test_targets = features[-400:], targets[-400:]
data_features, data_targets = features[:-400], targets[:-400]

# Hold out the last 400 of the remaining data as a validation set
val_features, val_targets = data_features[-400:], data_targets[-400:]
train_features, train_targets = data_features[:-400], data_targets[:-400]
```


```python
# Head of the scaled and projected features
features.head()
```

## Build the network

The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.

We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.
  


```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)# signals into hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs)# signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)# signals into final output layer
        final_outputs = final_inputs# signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error
        output_errors = targets - final_outputs # Output layer error is the difference between desired target and actual output.
        output_grad = output_errors
        
        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_grad)# errors propagated to the hidden layer
        hidden_grad = hidden_outputs * (1 - hidden_outputs)# hidden layer gradients
        
        # TODO: Update the weights
        # import pdb; pdb.set_trace()
        self.weights_hidden_to_output += np.dot(output_grad, hidden_outputs.T) * self.lr# update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += np.dot(hidden_errors * hidden_grad, inputs.T) * self.lr# update input-to-hidden weights with gradient descent step
         
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = inputs# signals into hidden layer
        hidden_outputs = self.sigmoid(np.dot(self.weights_input_to_hidden, hidden_inputs))# signals from hidden layer
        
        # TODO: Output layer
        final_inputs = hidden_outputs# signals into final output layer
        final_outputs = np.dot(self.weights_hidden_to_output, final_inputs)# signals from final output layer 
        
        return final_outputs
```


```python
def MSE(y, Y):
    return np.mean((y-Y)**2)
```

## Training the network

Set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.

We use a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. More training passes could also be used than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.

### Choose the number of epochs
This is the number of times the dataset will pass through the network, each time updating the weights. As the number of epochs increases, the network becomes better and better at predicting the targets in the training set. You'll need to choose enough epochs to train the network well but not too many or you'll be overfitting.

### Choose the learning rate
This scales the size of weight updates. If this is too big, the weights tend to explode and the network fails to fit the data. A good choice to start at is 0.1. If the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.

### Choose the number of hidden nodes
The more hidden nodes you have, the more accurate predictions the model will make. Try a few different numbers and see how it affects the performance. You can look at the losses dictionary for a metric of the network performance. If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.


```python
import sys

### Set the hyperparameters here ###
epochs = 1000

### It's better got set no larger than 0.01 if the features are scaled
learning_rate = 0.0004
hidden_nodes = 300
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]):
        network.train(record, target)
    
    # Printing out the training progress
    # import pdb; pdb.set_trace()
    train_loss = MSE(network.run(train_features), train_targets.values)
    val_loss = MSE(network.run(val_features), val_targets.values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
```


```python
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymin=0.5, ymax=1.5)
```




    (0.5, 1.5)




![png](output_17_1.png)



```python
fig, ax = plt.subplots(figsize=(16,4))

# mean, std = scales_target
# predictions = network.run(train_features[0:400])*std + mean
predictions = network.run(train_features[0:400]).T
ax.plot(predictions[:,0], label='Prediction')

# ax.plot((train_targets[0:2000]*std + mean).values, label='Data')
ax.plot(train_targets[0:400].values, label='Ground Truth')
ax.set_xlim(right=len(predictions))
ax.legend()
```




    <matplotlib.legend.Legend at 0x25ed6ceb8>




![png](output_18_1.png)



```python
fig, ax = plt.subplots(figsize=(16,4))

# mean, std = scales_target
# predictions = network.run(val_features)*std + mean
predictions = network.run(val_features).T
ax.plot(predictions[:,0], label='Prediction')
# ax.plot((val_targets*std + mean).values, label='Data')

ax.plot(val_targets.values, label='Ground Truth')
ax.set_xlim(right=len(predictions))
ax.legend()
```




    <matplotlib.legend.Legend at 0x113ffafd0>




![png](output_19_1.png)


## Predictions on Testing data

Here, use the test data to view how well the network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.


```python
fig, ax = plt.subplots(figsize=(16,4))

# mean, std = scales_target
# predictions = network.run(test_features)*std + mean
predictions = network.run(test_features).T
ax.plot(predictions[:,0], label='Prediction')
# ax.plot((test_targets*std + mean).values, label='Data')

ax.plot(test_targets.values, label='Ground Truth')
ax.set_xlim(right=len(predictions))
ax.legend()
```




    <matplotlib.legend.Legend at 0x1142a5ba8>




![png](output_21_1.png)


## Unit tests

Run these unit tests to check the correctness of the network implementation.


```python
import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.sigmoid(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)
```

    .EF..
    ======================================================================
    ERROR: test_data_loaded (__main__.TestMethods)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "<ipython-input-15-1b584817c956>", line 21, in test_data_loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    NameError: name 'rides' is not defined
    
    ======================================================================
    FAIL: test_data_path (__main__.TestMethods)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "<ipython-input-15-1b584817c956>", line 17, in test_data_path
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
    AssertionError: False is not true
    
    ----------------------------------------------------------------------
    Ran 5 tests in 0.010s
    
    FAILED (failures=1, errors=1)





    <unittest.runner.TextTestResult run=5 errors=1 failures=1>




```python

```
