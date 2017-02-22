# Tools for csv Analysis and Regression Based on Deep Structures

In this project, 2 layer neural network is used for predicting funds index within a single csv file.

Here we set 'header' equals to None to indicate that there are no headers in the csv file, Here the index is int64 rather than string.
```python
raw_data = pd.read_csv(data_path, delimiter=',', header=None)
```

Get the target labels in the second column and use remaining columns as feature.
```python
targets = raw_data[2]
features = raw_data.drop([0, 2], axis=1)
```

## Splitting the data into training, testing, and validation sets

We'll save the last 21 days of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.

We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).

Save the last 200 as test data.
```python
test_features, test_targets = features[-200:], targets[-200:]
data_features, data_targets = features[:-200], targets[:-200]
```

Hold out the last 400 of the remaining data as a validation set.
```python
val_features, val_targets = data_features[-400:], data_targets[-400:]
train_features, train_targets = data_features[:-400], data_targets[:-400]
```

## Build the network

The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.

We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.

```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        ...
    def train(self, inputs_list, targets_list):
        ...
    def run(self, inputs_list):
        ...
```

## Training the network

Set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.

We use a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. More training passes could also be used than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.

## Author infos

### Basic info

Yida Wang

Email: yidawang.cn@gmail.com

### Publications

[ZigzagNet: Efficient Deep Learning for Real Object Recognition Based on 3D Models](https://www.researchgate.net/profile/Yida_Wang/publications?sorting=recentlyAdded)

[Self-restraint Object Recognition by Model Based CNN Learning](http://ieeexplore.ieee.org/document/7532438/)

[Face Recognition Using Local PCA Filters](http://link.springer.com/chapter/10.1007%2F978-3-319-25417-3_5)

[CNTK on Mac: 2D Object Restoration and Recognition Based on 3D Model](https://www.microsoft.com/en-us/research/academic-program/microsoft-open-source-challenge/)

[Large-Scale 3D Shape Retrieval from ShapeNet Core55](https://shapenet.cs.stanford.edu/shrec16/shrec16shapenet.pdf)

### Personal Links

[ResearchGate](https://www.researchgate.net/profile/Yida_Wang), [Github](https://github.com/wangyida), [GSoC 2016](https://summerofcode.withgoogle.com/archive/2016/projects/4623962327744512/), [GSoC 2015](https://www.google-melange.com/archive/gsoc/2015/orgs/opencv/projects/wangyida.html)
