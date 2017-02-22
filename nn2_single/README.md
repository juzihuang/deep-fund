
# One file training - initial tool for funds regression

In this project, 2 layer neural network is used for predicting funds index within a single csv file.




```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load and prepare the data

A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!


```python
# Here is where we get the raw data
data_path = '/Volumes/机器学习/发送文件/发送文件/1.csv'

# Here we set 'header' equals to None to indicate that there are no headers in the csv
# file, Here the index is int64 rather than string.
raw_data = pd.read_csv(data_path, delimiter=',', header=None)

# Get the target labels in the second column and use remaining columns as feature
targets = raw_data[2]
features = raw_data.drop([0, 2], axis=1)
```


```python
# Head of the features
features.head()
# target.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
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
      <th>756</th>
      <th>757</th>
      <th>758</th>
      <th>759</th>
      <th>760</th>
      <th>761</th>
      <th>762</th>
      <th>763</th>
      <th>764</th>
      <th>765</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.633065</td>
      <td>-0.008128</td>
      <td>0.045430</td>
      <td>-0.002063</td>
      <td>-0.013119</td>
      <td>-0.007422</td>
      <td>-0.006651</td>
      <td>-0.007802</td>
      <td>-0.006543</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.465103</td>
      <td>-0.011930</td>
      <td>0.018900</td>
      <td>-0.023502</td>
      <td>-0.027506</td>
      <td>-0.015769</td>
      <td>-0.018849</td>
      <td>-0.015969</td>
      <td>-0.018974</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.466022</td>
      <td>-0.012439</td>
      <td>-0.023620</td>
      <td>-0.027816</td>
      <td>-0.031575</td>
      <td>-0.024687</td>
      <td>-0.015381</td>
      <td>-0.019730</td>
      <td>-0.018320</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.288603</td>
      <td>-0.015906</td>
      <td>-0.042954</td>
      <td>-0.014294</td>
      <td>-0.008752</td>
      <td>-0.012817</td>
      <td>-0.008098</td>
      <td>-0.008985</td>
      <td>-0.006139</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.517867</td>
      <td>-0.007269</td>
      <td>-0.014323</td>
      <td>-0.038031</td>
      <td>-0.026898</td>
      <td>-0.025223</td>
      <td>-0.024893</td>
      <td>-0.015078</td>
      <td>-0.011522</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.821830</td>
      <td>0.042320</td>
      <td>-0.050873</td>
      <td>0.017128</td>
      <td>0.006856</td>
      <td>0.003431</td>
      <td>0.005129</td>
      <td>0.001168</td>
      <td>-0.002225</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.508287</td>
      <td>0.026798</td>
      <td>-0.026389</td>
      <td>-0.001554</td>
      <td>0.000258</td>
      <td>0.007578</td>
      <td>0.013382</td>
      <td>0.015420</td>
      <td>0.008161</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.431805</td>
      <td>-0.003498</td>
      <td>-0.029886</td>
      <td>-0.014871</td>
      <td>-0.017959</td>
      <td>-0.031987</td>
      <td>-0.030681</td>
      <td>-0.025694</td>
      <td>-0.022653</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>0.354146</td>
      <td>0.003070</td>
      <td>0.003510</td>
      <td>-0.007172</td>
      <td>-0.035731</td>
      <td>-0.029357</td>
      <td>-0.012840</td>
      <td>-0.001463</td>
      <td>-0.007849</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.502485</td>
      <td>0.007092</td>
      <td>-0.020059</td>
      <td>-0.007007</td>
      <td>-0.007578</td>
      <td>-0.024629</td>
      <td>-0.021615</td>
      <td>-0.011337</td>
      <td>-0.011230</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.562543</td>
      <td>0.069652</td>
      <td>-0.021062</td>
      <td>0.023777</td>
      <td>-0.002315</td>
      <td>0.006924</td>
      <td>0.012881</td>
      <td>0.015629</td>
      <td>0.019942</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>0.643843</td>
      <td>0.011303</td>
      <td>-0.034423</td>
      <td>0.010298</td>
      <td>-0.003053</td>
      <td>-0.005764</td>
      <td>-0.002780</td>
      <td>-0.003133</td>
      <td>-0.007062</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>0.691513</td>
      <td>0.015337</td>
      <td>-0.027689</td>
      <td>-0.009944</td>
      <td>-0.023894</td>
      <td>-0.028269</td>
      <td>-0.027326</td>
      <td>-0.017943</td>
      <td>-0.017572</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>0.272150</td>
      <td>0.004544</td>
      <td>-0.058745</td>
      <td>-0.001189</td>
      <td>-0.002648</td>
      <td>-0.007108</td>
      <td>-0.009490</td>
      <td>-0.011212</td>
      <td>-0.010307</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>0.499114</td>
      <td>0.044382</td>
      <td>-0.030476</td>
      <td>0.038933</td>
      <td>0.031592</td>
      <td>0.032043</td>
      <td>0.019703</td>
      <td>0.017137</td>
      <td>0.008338</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>0.691380</td>
      <td>0.042396</td>
      <td>-0.014040</td>
      <td>-0.004883</td>
      <td>-0.012518</td>
      <td>-0.012249</td>
      <td>-0.010022</td>
      <td>-0.008078</td>
      <td>-0.001491</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.0</td>
      <td>0.268631</td>
      <td>0.068462</td>
      <td>-0.021131</td>
      <td>0.019932</td>
      <td>0.017450</td>
      <td>0.008738</td>
      <td>0.013499</td>
      <td>0.003659</td>
      <td>0.003417</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>0.591808</td>
      <td>0.009512</td>
      <td>-0.013618</td>
      <td>0.012560</td>
      <td>0.009261</td>
      <td>0.000604</td>
      <td>0.000244</td>
      <td>0.000316</td>
      <td>-0.002771</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.0</td>
      <td>0.250000</td>
      <td>0.012709</td>
      <td>-0.004227</td>
      <td>-0.000447</td>
      <td>-0.003586</td>
      <td>-0.008172</td>
      <td>-0.006915</td>
      <td>-0.007365</td>
      <td>-0.007711</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>0.388441</td>
      <td>0.024113</td>
      <td>-0.044811</td>
      <td>0.009709</td>
      <td>-0.005288</td>
      <td>-0.002965</td>
      <td>-0.005244</td>
      <td>-0.006135</td>
      <td>-0.008778</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.0</td>
      <td>0.119539</td>
      <td>-0.021035</td>
      <td>-0.043767</td>
      <td>0.000157</td>
      <td>-0.013239</td>
      <td>-0.031578</td>
      <td>-0.028586</td>
      <td>-0.017796</td>
      <td>-0.026139</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
      <td>0.336139</td>
      <td>-0.004463</td>
      <td>0.043866</td>
      <td>0.004383</td>
      <td>0.004177</td>
      <td>-0.003094</td>
      <td>-0.002165</td>
      <td>-0.001797</td>
      <td>0.009702</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>0.287500</td>
      <td>0.134160</td>
      <td>-0.026385</td>
      <td>0.067074</td>
      <td>0.033118</td>
      <td>0.017310</td>
      <td>0.029247</td>
      <td>0.023876</td>
      <td>0.036840</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.0</td>
      <td>0.376689</td>
      <td>0.034131</td>
      <td>-0.055621</td>
      <td>-0.008201</td>
      <td>-0.018929</td>
      <td>-0.013773</td>
      <td>-0.019498</td>
      <td>-0.006390</td>
      <td>-0.006644</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>0.387274</td>
      <td>-0.003170</td>
      <td>0.030162</td>
      <td>-0.005236</td>
      <td>0.001009</td>
      <td>-0.001784</td>
      <td>-0.013321</td>
      <td>-0.014802</td>
      <td>-0.016370</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.0</td>
      <td>0.613768</td>
      <td>0.021752</td>
      <td>0.014239</td>
      <td>-0.003562</td>
      <td>-0.009641</td>
      <td>-0.014999</td>
      <td>-0.018945</td>
      <td>-0.006355</td>
      <td>0.000305</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.0</td>
      <td>0.468382</td>
      <td>-0.022885</td>
      <td>-0.000383</td>
      <td>-0.018507</td>
      <td>-0.027481</td>
      <td>-0.033638</td>
      <td>-0.033941</td>
      <td>-0.025358</td>
      <td>-0.029284</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>0.386798</td>
      <td>-0.009563</td>
      <td>0.001435</td>
      <td>0.000792</td>
      <td>-0.000482</td>
      <td>0.000282</td>
      <td>-0.002661</td>
      <td>0.000429</td>
      <td>0.000148</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
      <td>0.559159</td>
      <td>-0.023616</td>
      <td>-0.028476</td>
      <td>-0.015900</td>
      <td>-0.019624</td>
      <td>-0.019879</td>
      <td>-0.015371</td>
      <td>-0.009560</td>
      <td>-0.009978</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1.0</td>
      <td>0.573649</td>
      <td>-0.039832</td>
      <td>-0.035924</td>
      <td>-0.013402</td>
      <td>-0.011273</td>
      <td>-0.019818</td>
      <td>-0.023598</td>
      <td>-0.022807</td>
      <td>-0.018165</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2256</th>
      <td>1.0</td>
      <td>0.401222</td>
      <td>0.050471</td>
      <td>-0.118596</td>
      <td>0.012210</td>
      <td>0.003714</td>
      <td>0.000435</td>
      <td>-0.000021</td>
      <td>0.002824</td>
      <td>0.002879</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2257</th>
      <td>1.0</td>
      <td>0.250000</td>
      <td>0.035190</td>
      <td>0.022177</td>
      <td>0.034144</td>
      <td>0.016908</td>
      <td>0.013079</td>
      <td>0.011929</td>
      <td>0.012005</td>
      <td>0.005737</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2258</th>
      <td>1.0</td>
      <td>0.589855</td>
      <td>-0.016670</td>
      <td>-0.023450</td>
      <td>-0.006702</td>
      <td>-0.015718</td>
      <td>-0.012640</td>
      <td>-0.013510</td>
      <td>-0.013266</td>
      <td>-0.014948</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2259</th>
      <td>1.0</td>
      <td>0.250000</td>
      <td>0.015376</td>
      <td>-0.053459</td>
      <td>0.024916</td>
      <td>0.002897</td>
      <td>0.003285</td>
      <td>-0.000977</td>
      <td>-0.004020</td>
      <td>-0.005412</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2260</th>
      <td>1.0</td>
      <td>0.499907</td>
      <td>0.010882</td>
      <td>-0.014549</td>
      <td>-0.004860</td>
      <td>-0.014883</td>
      <td>-0.017199</td>
      <td>-0.012668</td>
      <td>-0.009403</td>
      <td>-0.007516</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2261</th>
      <td>1.0</td>
      <td>0.257143</td>
      <td>-0.021325</td>
      <td>-0.028458</td>
      <td>-0.038737</td>
      <td>-0.045476</td>
      <td>-0.068306</td>
      <td>-0.055274</td>
      <td>-0.044052</td>
      <td>-0.037541</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2262</th>
      <td>1.0</td>
      <td>0.525745</td>
      <td>0.023338</td>
      <td>-0.032309</td>
      <td>-0.004533</td>
      <td>-0.006891</td>
      <td>-0.007965</td>
      <td>-0.010770</td>
      <td>-0.009619</td>
      <td>-0.010406</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2263</th>
      <td>1.0</td>
      <td>0.262999</td>
      <td>0.000639</td>
      <td>0.014183</td>
      <td>-0.003318</td>
      <td>-0.008763</td>
      <td>-0.007404</td>
      <td>0.000029</td>
      <td>0.001922</td>
      <td>-0.014275</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2264</th>
      <td>1.0</td>
      <td>0.417132</td>
      <td>0.070527</td>
      <td>0.014642</td>
      <td>0.036960</td>
      <td>0.016851</td>
      <td>0.009730</td>
      <td>-0.010918</td>
      <td>-0.002937</td>
      <td>-0.011130</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2265</th>
      <td>1.0</td>
      <td>0.542188</td>
      <td>0.003777</td>
      <td>-0.024189</td>
      <td>0.005297</td>
      <td>0.002310</td>
      <td>-0.002572</td>
      <td>-0.005687</td>
      <td>0.000540</td>
      <td>-0.007077</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2266</th>
      <td>1.0</td>
      <td>0.301128</td>
      <td>0.065764</td>
      <td>0.058868</td>
      <td>0.028250</td>
      <td>0.010425</td>
      <td>0.007921</td>
      <td>0.034471</td>
      <td>0.035857</td>
      <td>0.027531</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2267</th>
      <td>1.0</td>
      <td>0.547685</td>
      <td>0.034580</td>
      <td>-0.008732</td>
      <td>0.014965</td>
      <td>0.005946</td>
      <td>0.003962</td>
      <td>0.008975</td>
      <td>0.001831</td>
      <td>0.000283</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2268</th>
      <td>1.0</td>
      <td>0.298980</td>
      <td>-0.020760</td>
      <td>0.022138</td>
      <td>-0.006044</td>
      <td>-0.024692</td>
      <td>-0.043903</td>
      <td>-0.022301</td>
      <td>-0.011454</td>
      <td>-0.024954</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2269</th>
      <td>1.0</td>
      <td>0.324217</td>
      <td>-0.021128</td>
      <td>0.041928</td>
      <td>-0.018727</td>
      <td>-0.055213</td>
      <td>-0.061420</td>
      <td>-0.049226</td>
      <td>-0.043390</td>
      <td>-0.040865</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2270</th>
      <td>1.0</td>
      <td>0.392755</td>
      <td>-0.028264</td>
      <td>-0.019663</td>
      <td>-0.000646</td>
      <td>0.001684</td>
      <td>0.004042</td>
      <td>0.014506</td>
      <td>-0.005975</td>
      <td>-0.003631</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>1.0</td>
      <td>0.374384</td>
      <td>-0.000729</td>
      <td>-0.109177</td>
      <td>-0.018989</td>
      <td>-0.011056</td>
      <td>-0.018789</td>
      <td>-0.017662</td>
      <td>-0.016723</td>
      <td>-0.000786</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2272</th>
      <td>1.0</td>
      <td>0.250000</td>
      <td>0.021453</td>
      <td>-0.080756</td>
      <td>0.004223</td>
      <td>-0.009143</td>
      <td>-0.009353</td>
      <td>-0.012321</td>
      <td>-0.006621</td>
      <td>-0.005522</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2273</th>
      <td>1.0</td>
      <td>0.326327</td>
      <td>0.111251</td>
      <td>-0.005398</td>
      <td>0.046113</td>
      <td>0.005466</td>
      <td>-0.005531</td>
      <td>-0.004479</td>
      <td>-0.005507</td>
      <td>-0.015124</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2274</th>
      <td>1.0</td>
      <td>0.385839</td>
      <td>-0.004221</td>
      <td>0.034052</td>
      <td>-0.028567</td>
      <td>-0.025439</td>
      <td>-0.025039</td>
      <td>-0.022334</td>
      <td>-0.020730</td>
      <td>-0.018105</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2275</th>
      <td>1.0</td>
      <td>0.588961</td>
      <td>0.021317</td>
      <td>0.042704</td>
      <td>-0.001158</td>
      <td>-0.030788</td>
      <td>-0.035663</td>
      <td>-0.032278</td>
      <td>-0.029840</td>
      <td>-0.036153</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2276</th>
      <td>1.0</td>
      <td>0.317914</td>
      <td>0.028936</td>
      <td>0.015046</td>
      <td>0.020958</td>
      <td>0.001856</td>
      <td>0.004623</td>
      <td>0.004917</td>
      <td>0.008279</td>
      <td>0.002178</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>1.0</td>
      <td>0.441557</td>
      <td>0.077732</td>
      <td>-0.031530</td>
      <td>0.029067</td>
      <td>0.019145</td>
      <td>0.022328</td>
      <td>0.034876</td>
      <td>0.037173</td>
      <td>0.033869</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2278</th>
      <td>1.0</td>
      <td>0.557235</td>
      <td>-0.008683</td>
      <td>0.007668</td>
      <td>-0.011973</td>
      <td>-0.040942</td>
      <td>-0.034763</td>
      <td>-0.028701</td>
      <td>-0.018362</td>
      <td>-0.022512</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2279</th>
      <td>1.0</td>
      <td>0.666628</td>
      <td>-0.001860</td>
      <td>-0.008274</td>
      <td>0.007063</td>
      <td>-0.007772</td>
      <td>-0.008884</td>
      <td>-0.015806</td>
      <td>-0.012222</td>
      <td>-0.011306</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2280</th>
      <td>1.0</td>
      <td>0.456716</td>
      <td>-0.011850</td>
      <td>-0.022237</td>
      <td>-0.032937</td>
      <td>-0.024335</td>
      <td>-0.034391</td>
      <td>-0.032627</td>
      <td>-0.030043</td>
      <td>-0.018637</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2281</th>
      <td>1.0</td>
      <td>0.384491</td>
      <td>-0.000486</td>
      <td>-0.002127</td>
      <td>-0.007177</td>
      <td>-0.010848</td>
      <td>-0.012048</td>
      <td>-0.006198</td>
      <td>0.000277</td>
      <td>-0.002455</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2282</th>
      <td>1.0</td>
      <td>0.699238</td>
      <td>0.062360</td>
      <td>0.047896</td>
      <td>0.019337</td>
      <td>-0.021258</td>
      <td>-0.021015</td>
      <td>-0.020084</td>
      <td>-0.025043</td>
      <td>-0.009836</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2283</th>
      <td>1.0</td>
      <td>0.365379</td>
      <td>-0.039267</td>
      <td>-0.009158</td>
      <td>-0.029373</td>
      <td>-0.010012</td>
      <td>-0.015078</td>
      <td>-0.014861</td>
      <td>-0.007860</td>
      <td>0.003692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2284</th>
      <td>1.0</td>
      <td>0.357303</td>
      <td>-0.010133</td>
      <td>0.012579</td>
      <td>-0.002434</td>
      <td>-0.002073</td>
      <td>0.001721</td>
      <td>0.001608</td>
      <td>-0.000941</td>
      <td>0.000254</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2285</th>
      <td>1.0</td>
      <td>0.250000</td>
      <td>-0.026411</td>
      <td>0.039313</td>
      <td>-0.015666</td>
      <td>0.000934</td>
      <td>-0.010864</td>
      <td>-0.007283</td>
      <td>-0.001779</td>
      <td>-0.005951</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2286 rows × 764 columns</p>
</div>



## Scaling target variables
To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

The scaling factors are saved so we can go backwards when we use the network for predictions.


```python
quant_features = ['1', '3', '4', '5', '6', '7']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-36-cfb6655dace1> in <module>()
          3 scaled_features = {}
          4 for each in quant_features:
    ----> 5     mean, std = data[each].mean(), data[each].std()
          6     scaled_features[each] = [mean, std]
          7     data.loc[:, each] = (data[each] - mean)/std


    NameError: name 'data' is not defined


## Splitting the data into training, testing, and validation sets

We'll save the last 21 days of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.

We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).


```python
# Save the last 200 as test data
test_features, test_targets = features[-200:], targets[-200:]
data_features, data_targets = features[:-200], targets[:-200]

# Hold out the last 400 of the remaining data as a validation set
val_features, val_targets = data_features[-400:], data_targets[-400:]
train_features, train_targets = data_features[:-400], data_targets[:-400]
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
epochs = 3000
learning_rate = 0.001
hidden_nodes = 100
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
    train_loss = MSE(network.run(train_features), train_targets.values)
    val_loss = MSE(network.run(val_features), val_targets.values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
```

    /usr/local/lib/python3.6/site-packages/ipykernel/__main__.py:18: RuntimeWarning: overflow encountered in exp


    Progress: 99.9% ... Training loss: 0.081 ... Validation loss: 0.093


```python
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)
```




    (0.0, 0.5)




![png](output_15_1.png)


## Check out the predictions

Here, use the test data to view how well the network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.


```python
fig, ax = plt.subplots(figsize=(8,4))

# mean, std = scaled_features['cnt']
# predictions = network.run(test_features)*std + mean
predictions = network.run(test_features).T
ax.plot(predictions[:,0], label='Prediction')
# ax.plot((test_targets*std + mean).values, label='Data')

ax.plot(test_targets.values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

# dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
# dates = dates.apply(lambda d: d.strftime('%b %d'))
# ax.set_xticks(np.arange(len(dates))[12::24])
# _ = ax.set_xticklabels(dates[12::24], rotation=45)
```

    /usr/local/lib/python3.6/site-packages/ipykernel/__main__.py:18: RuntimeWarning: overflow encountered in exp





    <matplotlib.legend.Legend at 0x113dedf28>




![png](output_17_2.png)



```python
predictions[:,0].shape
```




    (200,)



## Thinking about your results
 
Answer these questions about your results. How well does the model predict the data? Where does it fail? Why does it fail where it does?

> **Note:** You can edit the text in this cell by double clicking on it. When you want to render the text, press control + enter

#### Your answer below

## Unit tests

Run these unit tests to check the correctness of your network implementation. These tests must all be successful to pass the project.


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

    .....
    ----------------------------------------------------------------------
    Ran 5 tests in 0.023s
    
    OK





    <unittest.runner.TextTestResult run=5 errors=0 failures=0>




```python

```


```python

```


```python

```
