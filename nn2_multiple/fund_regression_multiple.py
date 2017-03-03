import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
             '/Volumes/机器学习/发送文件/发送文件/20.csv']

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
print('not a number is located there: ', np.where(np.isnan(features.values) == True))

U, s, Vt = np.linalg.svd(features.values, full_matrices=False)
V = Vt.T

# PCs are already sorted by descending order
# of the singular values (i.e. by the
# proportion of total variance they explain)

# Method 1 reconstruction:
# if we use all of the PCs we can reconstruct the noisy signal perfectly
S = np.diag(s)
Mhat = np.dot(U, np.dot(S, V.T))
# print "Using all PCs, MSE = %.6G" %(np.mean((M - Mhat)**2))

dim_remain = 200
# Method 2 weak reconstruction:
# if we use only the first few PCs the reconstruction is less accurate
Mhat2 = np.dot(U[:, :dim_remain],
               np.dot(S[:dim_remain, :dim_remain], V[:,:dim_remain].T))

# Method 3 dimention reduction:
# if we use only the first few PCs the reconstruction is less accurate
Mhat3 = np.dot(U[:, :dim_remain], S[:dim_remain, :dim_remain])
right = features
wrong = pd.DataFrame(Mhat)
print('not a number is located there: ', np.where(np.isnan(features.values) == True))
# print("Using first few PCs, MSE = %.6G" %(np.mean((M - Mhat2)**2)))

# Save the last 400 as test data
# Hold out the last 400 of the remaining data as a validation set
test_features, test_targets = features[-400:], targets[-400:]
data_features, data_targets = features[:-400], targets[:-400]

val_features, val_targets = data_features[-400:], data_targets[-400:]
train_features, train_targets = data_features[:-400], data_targets[:-400]

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

def MSE(y, Y):
    return np.mean((y-Y)**2)

import sys

### Set the hyperparameters here ###
epochs = 600

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
    import pdb; pdb.set_trace()
    train_loss = MSE(network.run(train_features), train_targets.values)
    val_loss = MSE(network.run(val_features), val_targets.values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
