## Testing MNIST dataset using Neural Network

### A few words about MNIST dataset:

*The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.(source - http://yann.lecun.com/exdb/mnist/)*

### Model summary

A CNN model is used to test the MNIST dataset and predict the value of the handwritten digit.The model is as given below:

| Layer   |   Input  |  Output  | Kernel size | Receptive field |
|:---------:|:--------:|:--------:|:-----------:|:---------------:|
| Conv1   |  28*28*1 | 26*26*14 |     3x3     |        3        |
| Conv2   | 26*26*14 | 24*24*30 |     3x3     |        5        |
| Pool1   | 24*24*30 | 12*12*30 |     2x2     |        10       |
| Conv3   | 12*12*30 | 12*12*14 |     1x1     |        10       |
| Conv4   | 12*12*14 | 10*10*30 |     3x3     |        12       |
| Conv5   | 10*10*30 |  8*8*39  |     3x3     |        14       |
| Conv6   |  8*8*39  |  8*8*10  |     1x1     |        14       |
| AvgPool |  8*8*10  |  1*1*10  |      -      |        14       |
 
### *The network is built to meet the following constraints*

### 99.4% validation accuracy
### Less than 20k Parameters
### Less than 20 Epochs
### No fully connected layer
