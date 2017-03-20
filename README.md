
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
import os
import matplotlib.pyplot as plt
```

## Define functions for file scanning


```python
class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix='.csv'):
        self.directory=directory
        self.prefix=prefix
        self.postfix=postfix

    def scan_files(self):
        files_list=[]

        for dirpath,dirnames,filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath,special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))

        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list
```

## Load and prepare the data

A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, I write the code to load and prepare the data.


```python
# Here is where we get the raw data

files_dir = '/Volumes/MachineFunds/data_3000'
file_scanner=ScanFile(files_dir)
data_list=file_scanner.scan_files()

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
targets = raw_data[1]

# the first column is really large which doesn't mean anything and the first second
# column is all the same which makes it impossible to get scaled
features = raw_data.drop([0, 1], axis=1)

# delete the raw data
del raw_data
```

    /usr/local/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


    Here we start to load all the data as listed below:
    File 1: /Volumes/MachineFunds/data_3000/1.csv
    File 2: /Volumes/MachineFunds/data_3000/10.csv
    File 3: /Volumes/MachineFunds/data_3000/11.csv
    File 4: /Volumes/MachineFunds/data_3000/12.csv
    File 5: /Volumes/MachineFunds/data_3000/13.csv
    File 6: /Volumes/MachineFunds/data_3000/14.csv
    File 7: /Volumes/MachineFunds/data_3000/15.csv
    File 8: /Volumes/MachineFunds/data_3000/16.csv
    File 9: /Volumes/MachineFunds/data_3000/17.csv
    File 10: /Volumes/MachineFunds/data_3000/18.csv
    File 11: /Volumes/MachineFunds/data_3000/19.csv
    File 12: /Volumes/MachineFunds/data_3000/2.csv
    File 13: /Volumes/MachineFunds/data_3000/20.csv
    File 14: /Volumes/MachineFunds/data_3000/21.csv
    File 15: /Volumes/MachineFunds/data_3000/22.csv
    File 16: /Volumes/MachineFunds/data_3000/23.csv
    File 17: /Volumes/MachineFunds/data_3000/24.csv
    File 18: /Volumes/MachineFunds/data_3000/25.csv
    File 19: /Volumes/MachineFunds/data_3000/26.csv
    File 20: /Volumes/MachineFunds/data_3000/27.csv
    File 21: /Volumes/MachineFunds/data_3000/28.csv
    File 22: /Volumes/MachineFunds/data_3000/29.csv
    File 23: /Volumes/MachineFunds/data_3000/3.csv
    File 24: /Volumes/MachineFunds/data_3000/30.csv
    File 25: /Volumes/MachineFunds/data_3000/31.csv
    File 26: /Volumes/MachineFunds/data_3000/32.csv
    File 27: /Volumes/MachineFunds/data_3000/33.csv
    File 28: /Volumes/MachineFunds/data_3000/34.csv
    File 29: /Volumes/MachineFunds/data_3000/35.csv
    File 30: /Volumes/MachineFunds/data_3000/36.csv
    File 31: /Volumes/MachineFunds/data_3000/37.csv
    File 32: /Volumes/MachineFunds/data_3000/38.csv
    File 33: /Volumes/MachineFunds/data_3000/39.csv
    File 34: /Volumes/MachineFunds/data_3000/4.csv
    File 35: /Volumes/MachineFunds/data_3000/40.csv
    File 36: /Volumes/MachineFunds/data_3000/41.csv
    File 37: /Volumes/MachineFunds/data_3000/42.csv
    File 38: /Volumes/MachineFunds/data_3000/43.csv
    File 39: /Volumes/MachineFunds/data_3000/44.csv
    File 40: /Volumes/MachineFunds/data_3000/45.csv
    File 41: /Volumes/MachineFunds/data_3000/46.csv
    File 42: /Volumes/MachineFunds/data_3000/47.csv
    File 43: /Volumes/MachineFunds/data_3000/48.csv
    File 44: /Volumes/MachineFunds/data_3000/49.csv
    File 45: /Volumes/MachineFunds/data_3000/5.csv
    File 46: /Volumes/MachineFunds/data_3000/50.csv
    File 47: /Volumes/MachineFunds/data_3000/6.csv
    File 48: /Volumes/MachineFunds/data_3000/7.csv
    File 49: /Volumes/MachineFunds/data_3000/8.csv
    File 50: /Volumes/MachineFunds/data_3000/9.csv


# Wash the raw data
Here we find some data in columns in form is not a number and delete those columns to further our experiments.


```python
# trace of the features
trace = np.linalg.matrix_rank(features.values)

# here is what we do to remove those columns with not a number values, column indexes
# are printed.
nan_col = np.where(np.isnan(features.values) == True)[1]
print('Not a Number is located there: ', nan_col)

# we drop those useless columns
features = features.dropna(axis=1)
```

    /usr/local/lib/python3.6/site-packages/numpy/linalg/linalg.py:1591: RuntimeWarning: invalid value encountered in greater
      return sum(S > tol)


    Not a Number is located there:  [2726 2727 2333 ..., 2771 2772 2773]



```python
# Head of the features
print(np.linalg.matrix_rank(features.values))
features.head(5)
```

    185





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>...</th>
      <th>2829</th>
      <th>2830</th>
      <th>2831</th>
      <th>2832</th>
      <th>2833</th>
      <th>2834</th>
      <th>2835</th>
      <th>2836</th>
      <th>2837</th>
      <th>2838</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.504336</td>
      <td>-0.001515</td>
      <td>-0.002974</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>-0.001515</td>
      <td>...</td>
      <td>-0.732213</td>
      <td>-0.321351</td>
      <td>-0.495196</td>
      <td>-0.528746</td>
      <td>-0.255171</td>
      <td>-1.354341</td>
      <td>-1.992957</td>
      <td>-1.667285</td>
      <td>-0.350672</td>
      <td>0.779335</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.731476</td>
      <td>0.001579</td>
      <td>-0.000677</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>0.001579</td>
      <td>...</td>
      <td>-0.850811</td>
      <td>-1.102105</td>
      <td>0.035168</td>
      <td>-0.417740</td>
      <td>-0.789885</td>
      <td>1.260397</td>
      <td>2.155116</td>
      <td>0.877689</td>
      <td>-0.739415</td>
      <td>0.535623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.557203</td>
      <td>-0.003547</td>
      <td>-0.006382</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>-0.003547</td>
      <td>...</td>
      <td>-1.065149</td>
      <td>-1.775345</td>
      <td>-1.896741</td>
      <td>-1.804887</td>
      <td>-2.080428</td>
      <td>-0.860390</td>
      <td>-0.898422</td>
      <td>-0.794951</td>
      <td>0.225396</td>
      <td>2.216433</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.393331</td>
      <td>-0.002041</td>
      <td>-0.006511</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>-0.002041</td>
      <td>...</td>
      <td>-0.068502</td>
      <td>-0.308363</td>
      <td>0.227580</td>
      <td>0.091325</td>
      <td>-0.112073</td>
      <td>-0.429996</td>
      <td>-0.306691</td>
      <td>-0.546743</td>
      <td>-0.233236</td>
      <td>1.740761</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.411687</td>
      <td>-0.002323</td>
      <td>-0.013646</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>-0.002323</td>
      <td>...</td>
      <td>-2.018887</td>
      <td>1.054963</td>
      <td>1.513833</td>
      <td>1.990833</td>
      <td>1.404082</td>
      <td>-1.848934</td>
      <td>-1.484598</td>
      <td>-1.179912</td>
      <td>-0.338743</td>
      <td>2.113630</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2647 columns</p>
</div>



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
print(np.linalg.matrix_rank(features.values))
```

    2130



```python
features.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>...</th>
      <th>2829</th>
      <th>2830</th>
      <th>2831</th>
      <th>2832</th>
      <th>2833</th>
      <th>2834</th>
      <th>2835</th>
      <th>2836</th>
      <th>2837</th>
      <th>2838</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.286760</td>
      <td>0.186127</td>
      <td>1.913584</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>0.186127</td>
      <td>...</td>
      <td>-0.635627</td>
      <td>-0.337530</td>
      <td>-0.503343</td>
      <td>-0.540678</td>
      <td>-0.295757</td>
      <td>-0.823949</td>
      <td>-1.222380</td>
      <td>-0.999809</td>
      <td>-0.393728</td>
      <td>-0.602980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.473115</td>
      <td>1.323473</td>
      <td>2.325315</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>1.323473</td>
      <td>...</td>
      <td>-0.725015</td>
      <td>-0.964377</td>
      <td>-0.047797</td>
      <td>-0.443712</td>
      <td>-0.776997</td>
      <td>1.011184</td>
      <td>1.431276</td>
      <td>0.497187</td>
      <td>-0.853929</td>
      <td>-0.662278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.562886</td>
      <td>-0.560357</td>
      <td>1.302992</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>-0.560357</td>
      <td>...</td>
      <td>-0.886563</td>
      <td>-1.504903</td>
      <td>-1.707172</td>
      <td>-1.655406</td>
      <td>-1.938482</td>
      <td>-0.477273</td>
      <td>-0.522171</td>
      <td>-0.486687</td>
      <td>0.288232</td>
      <td>-0.253319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.293016</td>
      <td>-0.006930</td>
      <td>1.279937</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>-0.006930</td>
      <td>...</td>
      <td>-0.135385</td>
      <td>-0.327103</td>
      <td>0.117471</td>
      <td>0.000964</td>
      <td>-0.166969</td>
      <td>-0.175205</td>
      <td>-0.143621</td>
      <td>-0.340688</td>
      <td>-0.254705</td>
      <td>-0.369055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.197140</td>
      <td>-0.110673</td>
      <td>0.001297</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>-0.110673</td>
      <td>...</td>
      <td>-1.605399</td>
      <td>0.767476</td>
      <td>1.222272</td>
      <td>1.660214</td>
      <td>1.197565</td>
      <td>-1.171076</td>
      <td>-0.897166</td>
      <td>-0.713128</td>
      <td>-0.379606</td>
      <td>-0.278332</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2647 columns</p>
</div>



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

    Using all PCs, MSE = 5.76234E-29
    Not a Number is located there:  (array([], dtype=int64), array([], dtype=int64))
    Using first few PCs, MSE = 0.00325042


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




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>690</th>
      <th>691</th>
      <th>692</th>
      <th>693</th>
      <th>694</th>
      <th>695</th>
      <th>696</th>
      <th>697</th>
      <th>698</th>
      <th>699</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-6.566169</td>
      <td>19.032715</td>
      <td>6.224122</td>
      <td>17.110095</td>
      <td>-0.085448</td>
      <td>4.473312</td>
      <td>5.887049</td>
      <td>14.041211</td>
      <td>6.691520</td>
      <td>14.901784</td>
      <td>...</td>
      <td>-0.309276</td>
      <td>0.141073</td>
      <td>0.013992</td>
      <td>-0.054940</td>
      <td>-0.117338</td>
      <td>-0.524727</td>
      <td>0.264559</td>
      <td>0.039653</td>
      <td>-0.330006</td>
      <td>0.076570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-20.884434</td>
      <td>26.640381</td>
      <td>-4.646190</td>
      <td>6.734236</td>
      <td>4.569623</td>
      <td>-10.895731</td>
      <td>1.875132</td>
      <td>13.964803</td>
      <td>20.332528</td>
      <td>17.547754</td>
      <td>...</td>
      <td>-0.134115</td>
      <td>-0.141545</td>
      <td>0.021505</td>
      <td>-0.398683</td>
      <td>-0.461437</td>
      <td>-0.430340</td>
      <td>0.639124</td>
      <td>0.289004</td>
      <td>0.070453</td>
      <td>0.302907</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.811612</td>
      <td>3.632730</td>
      <td>31.646053</td>
      <td>2.315861</td>
      <td>1.891268</td>
      <td>-7.555630</td>
      <td>16.222121</td>
      <td>3.621030</td>
      <td>18.086279</td>
      <td>7.869595</td>
      <td>...</td>
      <td>-0.083437</td>
      <td>0.155509</td>
      <td>-0.232784</td>
      <td>-0.195763</td>
      <td>-0.195178</td>
      <td>-0.124140</td>
      <td>0.406470</td>
      <td>-0.292933</td>
      <td>0.224434</td>
      <td>-0.016303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-8.083562</td>
      <td>-10.692772</td>
      <td>27.459058</td>
      <td>8.842852</td>
      <td>0.847682</td>
      <td>0.076430</td>
      <td>1.001536</td>
      <td>7.829298</td>
      <td>3.926591</td>
      <td>7.476953</td>
      <td>...</td>
      <td>0.052880</td>
      <td>0.042022</td>
      <td>0.078435</td>
      <td>0.217511</td>
      <td>-0.064960</td>
      <td>0.147006</td>
      <td>0.062521</td>
      <td>0.294573</td>
      <td>0.189536</td>
      <td>0.115230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.765388</td>
      <td>-19.099900</td>
      <td>31.664797</td>
      <td>10.331895</td>
      <td>-9.720690</td>
      <td>-10.548815</td>
      <td>-9.311432</td>
      <td>-5.036263</td>
      <td>10.492547</td>
      <td>0.172944</td>
      <td>...</td>
      <td>-0.049919</td>
      <td>0.035408</td>
      <td>-0.262899</td>
      <td>-0.175371</td>
      <td>-0.050954</td>
      <td>-0.291492</td>
      <td>0.127735</td>
      <td>0.151154</td>
      <td>-0.102863</td>
      <td>-0.065052</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 700 columns</p>
</div>



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
epochs = 2000

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

    Progress: 4.7% ... Training loss: 4.438 ... Validation loss: 5.2862

    /usr/local/lib/python3.6/site-packages/ipykernel/__main__.py:18: RuntimeWarning: overflow encountered in exp


    Progress: 99.9% ... Training loss: 1.063 ... Validation loss: 1.315

### plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymin=0.5, ymax=1.5)


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




    <matplotlib.legend.Legend at 0x10dd0a780>




![png](pca_nn_multiple/output_23_1.png)



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




    <matplotlib.legend.Legend at 0x109933ba8>




![png](pca_nn_multiple/output_24_1.png)


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




    <matplotlib.legend.Legend at 0x10a6e4c50>




![png](pca_nn_multiple/output_26_1.png)


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
