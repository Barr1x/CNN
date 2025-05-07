"""
Fall 2023, 10-417/617
Assignment-2
Programming - CNN
TAs in charge: Jared Mejia, Kaiwen Geng

IMPORTANT:
    DO NOT change any function signatures but feel free to add instance variables and methods to the classes.

October 2023
"""

from re import L
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import im2col_jhelper  # uncomment this line if you wish to make use of the im2col_helper.pyc file for experiments

CLASS_IDS = {
 'cat': 0,
 'dog': 1,
 'car': 2,
 'bus': 3,
 'train': 4,
 'boat': 5,
 'ball': 6,
 'pizza': 7,
 'chair': 8,
 'table': 9
 }

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    N, C, H, W = X.shape
    out_height = (H - k_height + 2 * padding) // stride + 1
    out_width = (W - k_width + 2 * padding) // stride + 1
    X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), mode = 'constant')
    
    col_matrix = np.zeros((C*k_height*k_width, out_height*out_width*N))
    for h in range(out_height):
        for w in range(out_width):
            for n in range(N):
                patches = X_padded[n, :, (h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)]
                patches = patches.reshape(C*k_height*k_width, 1)
                col_matrix[:, n + N*(w + out_width*h)] = patches[:,0]
    return col_matrix

    

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape

    Note: You must implement im2col yourself. If you use any functions from im2col_helper, you will lose 50
    points on this assignment.
    '''
    N, C, H, W = X_shape
    X_with_padding = np.zeros((N, C, H+2*padding, W+2*padding))
    X = np.zeros((N, C, H, W))
    out_height = (H - k_height + 2 * padding) // stride + 1
    out_width = (W - k_width + 2 * padding) // stride + 1
    for h in range(out_height):
        for w in range(out_width):
            for n in range(N):
                X_with_padding[n, :, (h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)] += grad_X_col[:, n + N*(w + out_width*h)].reshape(C, k_height, k_width)
    X = X_with_padding[:, :, padding:(H+padding), padding:(W+padding)]
    return X
    


class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x):
        self.x = x
        output = np.maximum(0,x)
        return output

    def backward(self, grad_wrt_out):
        grad_wrt_x = grad_wrt_out * (self.x > 0)
        return grad_wrt_x
    

class Dropout(Transform):
    """
    Implement this class. You may use your implementation from HW1
    """

    def __init__(self, p=0.1):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p = p
        

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        self.mask = np.ones(x.shape)
        if train:
            if self.p > 0:
                self.mask = np.random.binomial(1, 1-self.p, size = x.shape)
                return x * self.mask
        else:
            return x * (1 - self.p)  
        return x
    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        return grad_wrt_out * self.mask
    


class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        return dloss.reshape(self.shape)


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        
        C, im_height, im_width = input_shape
        num_filters, filter_height, filter_width = filter_shape
        b = np.sqrt(6)/np.sqrt((num_filters + C)*filter_height*filter_width)
        self.weights = np.random.uniform(-b, b, (num_filters, C, filter_height, filter_width))
        self.biases = np.zeros((num_filters, 1))
        self.flatten = Flatten()
        self.momentum_w = np.zeros(self.weights.shape)
        self.momentum_b = np.zeros(self.biases.shape)
        
    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        we recommend you use im2col here
        """
        self.pad = pad
        self.stride = stride
        self.x = inputs
        self.N, C, H, W = inputs.shape
        num_filters, filter_height, filter_width = self.weights.shape[0], self.weights.shape[2], self.weights.shape[3]
        out_height = (H - filter_height + 2 * pad) // stride + 1
        out_width = (W - filter_width + 2 * pad) // stride + 1
        self.X_col = im2col(inputs, filter_height, filter_width, pad, stride)
        self.W_col = self.flatten.forward(self.weights)
        out = (self.W_col @ self.X_col) + self.biases
        out = out.reshape(num_filters, out_height, out_width, self.N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        reshaped_dloss = dloss.transpose(1, 2, 3, 0).reshape(self.weights.shape[0], -1)
        self.grad_w = reshaped_dloss @ self.X_col.T
        self.gradient_wrt_weights= self.grad_w.reshape(self.weights.shape)
        self.grad_b = np.sum(reshaped_dloss, axis = 1, keepdims = True)
        self.gradient_wrt_biases = self.grad_b
        self.grad_x_col = self.W_col.T @ reshaped_dloss
        self.gradient_wrt_input = im2col_bw(self.grad_x_col, self.x.shape, self.weights.shape[2], self.weights.shape[3], self.pad, self.stride)
        return [self.gradient_wrt_weights, self.gradient_wrt_biases, self.gradient_wrt_input]
        

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Here we divide gradients by batch_size.
        """
        self.momentum_w = momentum_coeff * self.momentum_w + self.gradient_wrt_weights/self.N
        self.momentum_b = momentum_coeff * self.momentum_b + self.gradient_wrt_biases/self.N
        self.weights -= learning_rate * self.momentum_w
        self.biases -= learning_rate * self.momentum_b
        
        

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.weights, self.biases


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_height, self.filter_width = filter_shape
        self.stride = stride
        

    def forward(self, inputs):
        """
        forward pass of MaxPool
        inputs: (batch_size, C, H, W)
        """
        out_height = (inputs.shape[2] - self.filter_height) // self.stride + 1
        out_width = (inputs.shape[3] - self.filter_width) // self.stride + 1
        self.x = inputs
        output = np.zeros((inputs.shape[0], inputs.shape[1], out_height, out_width))
        for h in range(out_height):
            for w in range(out_width):
                output[:, :, h, w] = np.max(inputs[:, :, (h*self.stride):(h*self.stride+self.filter_height), (w*self.stride):(w*self.stride+self.filter_width)], axis = (2, 3))
        return output

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        grad_wrt_x = np.zeros(self.x.shape)
        out_height = (self.x.shape[2] - self.filter_height) // self.stride + 1
        out_width = (self.x.shape[3] - self.filter_width) // self.stride + 1
        for h in range(out_height):
            for w in range(out_width):
                for n in range(self.x.shape[0]):
                    for c in range(self.x.shape[1]):
                        max_index = np.argmax(self.x[n, c, (h*self.stride):(h*self.stride+self.filter_height), (w*self.stride):(w*self.stride+self.filter_width)])
                        max_index = np.unravel_index(max_index, (self.filter_height, self.filter_width))
                        grad_wrt_x[n, c, (h*self.stride):(h*self.stride+self.filter_height), (w*self.stride):(w*self.stride+self.filter_width)][max_index] = dloss[n, c, h, w]
        return grad_wrt_x
        



class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        b = np.sqrt(6)/np.sqrt(indim + outdim)
        self.weights = np.random.uniform(-b, b, (indim, outdim))
        self.biases = np.zeros((outdim, 1))
        self.momentum_w = np.zeros(self.weights.shape)
        self.momentum_b = np.zeros(self.biases.shape)
        

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.x = inputs
        return np.matmul(self.x, self.weights) + self.biases.T

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        self.grad_wrt_W = np.matmul(self.x.T, dloss)
        self.grad_wrt_b = np.sum(dloss, axis = 0, keepdims = True).T
        self.grad_wrt_input = np.matmul(dloss, self.weights.T)
        return [self.grad_wrt_W, self.grad_wrt_b, self.grad_wrt_input]

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.momentum_w = momentum_coeff * self.momentum_w + self.grad_wrt_W/self.x.shape[0]
        self.momentum_b = momentum_coeff * self.momentum_b + self.grad_wrt_b/self.x.shape[0]
        self.weights -= learning_rate * self.momentum_w
        self.biases -= learning_rate * self.momentum_b

    def get_wb_fc(self):
        """
        Return weights and biases as a tuple
        """
        return self.weights, self.biases


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in  the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        self.labels = labels
        softmax_numerator = np.exp(logits)
        softmax_denominator = np.sum(softmax_numerator, axis=1, keepdims=True)
        self.softmax = softmax_numerator/softmax_denominator
        cross_entropy = -np.sum(labels*np.log(self.softmax), axis=1, keepdims=True)
        cross_entropy = np.mean(cross_entropy)
        return cross_entropy


    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        grad_wrt_logits = self.softmax - self.labels
        return grad_wrt_logits

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        predicted_labels = np.argmax(self.softmax, axis=1)
        true_labels = np.argmax(self.labels, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy, predicted_labels
        

class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, dropout_probability=0):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        self.flatten = Flatten()
        self.Conv_layer = Conv((3, 32, 32), (1, 5, 5))
        self.ReLU_layer = ReLU()
        self.Dropout_layer = Dropout(dropout_probability)
        self.MaxPool_layer = MaxPool((2, 2), 2)
        self.Linear_layer = LinearLayer(1*16*16, 10)
        self.SoftMaxCrossEntropyLoss_layer = SoftMaxCrossEntropyLoss()
        


    def forward(self, inputs, y_labels, train = True):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        z = self.Conv_layer.forward(inputs)
        z = self.ReLU_layer.forward(z)
        z = self.Dropout_layer.forward(z, train)
        z = self.MaxPool_layer.forward(z)
        z = self.flatten.forward(z)
        z = self.Linear_layer.forward(z)
        loss = self.SoftMaxCrossEntropyLoss_layer.forward(z, y_labels)
        accuracy, predicted_labels = self.SoftMaxCrossEntropyLoss_layer.getAccu()
        return loss, accuracy, predicted_labels
        


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad_wrt_logits = self.SoftMaxCrossEntropyLoss_layer.backward()
        _, _, grad_wrt_input = self.Linear_layer.backward(grad_wrt_logits)
        grad_wrt_input = self.flatten.backward(grad_wrt_input)
        grad_wrt_input = self.MaxPool_layer.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer.backward(grad_wrt_input)
        grad_wrt_input = self.ReLU_layer.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Conv_layer.backward(grad_wrt_input)
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.Conv_layer.update(learning_rate, momentum_coeff)
        self.Linear_layer.update(learning_rate, momentum_coeff)

class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        My current best structure is as follows
        Conv of input shape 3x32x32 with filter size of 16x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then initialize first linear layer with hidden layer size of 64 neurons
        then apply Relu
        then apply Dropout with probability 0.5
        then initialize second linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        
        
        """
        self.flatten = Flatten()
        self.Convolution_layer1 = Conv((3, 32, 32), (16, 3, 3))
        self.ReLu_layer1 = ReLU()
        self.MaxPool_layer1 = MaxPool((2, 2), 2)
        self.Dropout_layer1 = Dropout(0.1)
        self.Convolution_layer2 = Conv((16, 17, 17), (16, 3, 3))
        self.ReLu_layer2 = ReLU()
        self.MaxPool_layer2 = MaxPool((2, 2), 2)
        self.Dropout_layer2 = Dropout(0.1)
        self.Linear_layer1 = LinearLayer(16*9*9, 64)
        self.ReLu_layer3 = ReLU()
        self.Dropout_layer3 = Dropout(0.5)
        self.Linear_layer2 = LinearLayer(64, 10)
        self.SoftMaxCrossEntropyLoss_layer =SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels, train = True):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        z = self.Convolution_layer1.forward(inputs)
        z = self.ReLu_layer1.forward(z)
        z = self.MaxPool_layer1.forward(z)
        z = self.Dropout_layer1.forward(z, train)
        z = self.Convolution_layer2.forward(z)
        z = self.ReLu_layer2.forward(z)
        z = self.MaxPool_layer2.forward(z)
        z = self.Dropout_layer2.forward(z, train)
        z = self.flatten.forward(z)
        z = self.Linear_layer1.forward(z)
        z = self.ReLu_layer3.forward(z)
        z = self.Dropout_layer3.forward(z, train)
        z = self.Linear_layer2.forward(z)
        loss = self.SoftMaxCrossEntropyLoss_layer.forward(z, y_labels)
        accuracy, predicted_labels = self.SoftMaxCrossEntropyLoss_layer.getAccu()
        return loss, accuracy, predicted_labels


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad_wrt_logits = self.SoftMaxCrossEntropyLoss_layer.backward()
        _, _, grad_wrt_input = self.Linear_layer2.backward(grad_wrt_logits)
        grad_wrt_input = self.Dropout_layer3.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer3.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Linear_layer1.backward(grad_wrt_input)
        grad_wrt_input = self.flatten.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.MaxPool_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer2.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Convolution_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer1.backward(grad_wrt_input)
        grad_wrt_input = self.MaxPool_layer1.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer1.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Convolution_layer1.backward(grad_wrt_input)
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.Convolution_layer1.update(learning_rate, momentum_coeff)
        self.Convolution_layer2.update(learning_rate, momentum_coeff)
        self.Linear_layer1.update(learning_rate, momentum_coeff)
        self.Linear_layer2.update(learning_rate, momentum_coeff)

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    (Conv -> Relu -> MaxPool -> Dropout)x3 -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, dropout_probability=0.1):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU, Conv, ReLU, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then apply Dropout with probability 0.1
        then Conv with filter size of 16 filters of size 3x3
        then apply Relu 
        then apply Dropout with probability 0.1
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """
        self.flatten = Flatten()
        self.Convolution_layer1 = Conv((3, 32, 32), (16, 3, 3))
        self.ReLu_layer1 = ReLU()
        self.MaxPool_layer1 = MaxPool((2, 2), 2)
        self.Dropout_layer1 = Dropout(dropout_probability)
        self.Convolution_layer2 = Conv((16, 17, 17), (16, 3, 3))
        self.ReLu_layer2 = ReLU()
        self.MaxPool_layer2 = MaxPool((2, 2), 2)
        self.Dropout_layer2 = Dropout(dropout_probability)
        self.Convolution_layer3 = Conv((16, 9, 9), (16, 3, 3))
        self.ReLu_layer3 = ReLU()
        self.Dropout_layer3 = Dropout(dropout_probability)
        self.Linear_layer = LinearLayer(16*11*11, 10)
        self.SoftMaxCrossEntropyLoss_layer =SoftMaxCrossEntropyLoss()
        


    def forward(self, inputs, y_labels, train = True):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        z = self.Convolution_layer1.forward(inputs)
        z = self.ReLu_layer1.forward(z)
        z = self.MaxPool_layer1.forward(z)
        z = self.Dropout_layer1.forward(z, train)
        z = self.Convolution_layer2.forward(z)
        z = self.ReLu_layer2.forward(z)
        z = self.MaxPool_layer2.forward(z)
        z = self.Dropout_layer2.forward(z, train)
        z = self.Convolution_layer3.forward(z)
        z = self.ReLu_layer3.forward(z)
        z = self.Dropout_layer3.forward(z, train)
        z = self.flatten.forward(z)
        z = self.Linear_layer.forward(z)
        loss = self.SoftMaxCrossEntropyLoss_layer.forward(z, y_labels)
        accuracy, predicted_labels = self.SoftMaxCrossEntropyLoss_layer.getAccu()
        return loss, accuracy, predicted_labels


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad_wrt_logits = self.SoftMaxCrossEntropyLoss_layer.backward()
        _, _, grad_wrt_input = self.Linear_layer.backward(grad_wrt_logits)
        grad_wrt_input = self.flatten.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer3.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer3.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Convolution_layer3.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.MaxPool_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer2.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Convolution_layer2.backward(grad_wrt_input)
        grad_wrt_input = self.Dropout_layer1.backward(grad_wrt_input)
        grad_wrt_input = self.MaxPool_layer1.backward(grad_wrt_input)
        grad_wrt_input = self.ReLu_layer1.backward(grad_wrt_input)
        _, _, grad_wrt_input = self.Convolution_layer1.backward(grad_wrt_input)
       

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.Convolution_layer1.update(learning_rate, momentum_coeff)
        self.Convolution_layer2.update(learning_rate, momentum_coeff)
        self.Convolution_layer3.update(learning_rate, momentum_coeff)
        self.Linear_layer.update(learning_rate, momentum_coeff)


def one_hot_encode(labels):
    """
    One hot encode labels
    """
    one_hot_labels = np.array([[i==label for i in range(len(CLASS_IDS.keys()))] for label in labels], np.int32)
    return one_hot_labels


def prep_imagenet_data(train_images, train_labels, val_images, val_labels):
    # onehot encode labels
    train_labels = one_hot_encode(train_labels)
    val_labels = one_hot_encode(val_labels)

    # standardize to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    val_images = (val_images - 127.5) / 127.5

    # put channels first
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    val_images = np.transpose(val_images, (0, 3, 1, 2))

    return train_images, train_labels, val_images, val_labels

def image_augmentation(images, labels):
    """
    Implement image augmentation here
    """
    N, C, H, W = images.shape
    new_images = np.zeros((N, C, H, W))
    new_labels = np.zeros((N, 10))
    for i in range(N):
        new_images[i] = images[i]
        new_labels[i] = labels[i]
        if np.random.uniform() > 0.5:
            new_images[i] = np.flip(new_images[i], axis = 1)
        if np.random.uniform() > 0.5:
            new_images[i] = np.flip(new_images[i], axis = 2)
        if np.random.uniform() > 0.5:
            brightness_factor = np.random.uniform(0.7, 1.0)
            new_images[i] *= brightness_factor
        
        
    total_images = np.concatenate((images, new_images), axis = 0)
    total_labels = np.concatenate((labels, new_labels), axis = 0)
    return total_images, total_labels


# Implement the training as you wish. This part will not be autograded
# Feel free to implement other helper libraries (i.e. matplotlib, seaborn) but please do not import other libraries (i.e. torch, tensorflow, etc.) for the training
#Note: make sure to download the data from the resources tab on piazza
if __name__ == '__main__':
    # This part may be helpful to write the training loop
   
    # # Training parameters
    # parser = argparse.ArgumentParser(description='CNN')
    # parser.add_argument('--batch_size', type=int, default = 128)
    # parser.add_argument('--learning_rate', type=float, default = 0.001)
    # parser.add_argument('--momentum', type=float, default = 0.95)
    # parser.add_argument('--num_epochs', type=int, default = 50)
    # parser.add_argument('--seed', type=int, default = 47)
    # parser.add_argument('--dropout_p', type=float, default=0.1)
    # parser.add_argument('--name_prefix', type=str, default=None)
    # parser.add_argument('--num_filters', type=int, default=16)
    # parser.add_argument('--filter_size', type=int, default=3)
    # args = parser.parse_args()
    # BATCH_SIZE = args.batch_size
    # LEARNING_RATE = args.learning_rate
    # MOMENTUM = args.momentum
    # EPOCHS = args.num_epochs
    # SEED = args.seed
    # print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
        
    def train(model, train_images, train_labels, val_images, val_labels, num_epochs=100, batch_size=128, lr=0.01, momentum=0.9):
        """
        Implement training loop here
        """
        num_samples = train_images.shape[0]
        num_batches = num_samples // batch_size
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            permutation = np.random.permutation(num_samples)
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]
            for i in range(num_batches):
                batch_images = train_images[i*batch_size:(i+1)*batch_size]
                batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
                model.forward(batch_images, batch_labels, train=True)
                model.backward()
                model.update(lr, momentum)
            
            train_l, train_accu, train_y_hat = model.forward(train_images, train_labels, train=False)
            val_l, val_accu, val_y_hat = model.forward(val_images, val_labels, train=False)
            train_loss.append(train_l)
            val_loss.append(val_l)
            train_accuracy.append(train_accu)
            val_accuracy.append(val_accu)

            print("Epoch: ", epoch, "Train Loss: ", train_l, "Train accuracy: ", train_accu)
            print("Epoch: ", epoch, "Val Loss: ", val_l, "Val accuracy: ", val_accu)
            
            if val_accu > best_val_accuracy:
                best_val_accuracy = val_accu
                best_model = model
                with open("model_checkpoint.pkl", "wb") as f:
                    pickle.dump(best_model, f)
        return train_loss, val_loss, train_accuracy, val_accuracy
    
    def plot(train_loss, val_loss, train_accuracy, val_accuracy):
        plt.plot(train_loss, label = "train loss")
        plt.plot(val_loss, label = "val loss")
        plt.legend()
        plt.show()
        plt.plot(train_accuracy, label = "train accuracy")
        plt.plot(val_accuracy, label = "val accuracy")
        plt.legend()
        plt.show()
            
    ## DATA EXPLORATION 
    with open("10417-tiny-imagenet-train-bal.pkl", "rb") as f:
        train_dict = pickle.load(f)
        train_images = train_dict["images"]
        train_labels = train_dict["labels"]
    
    with open("10417-tiny-imagenet-val-bal.pkl", "rb") as f:
        val_dict = pickle.load(f)
        val_images = val_dict["images"]
        val_images_copy = val_images
        val_labels = val_dict["labels"]
        
    
    ## Problem 1a: Data Vis
    # TODO: plot data samples for train/val
    def data_vis(data, tlabels, flabels, num_samples):
        fig, ax = plt.subplots(1, num_samples, figsize = (10, 10))
        for i in range(num_samples):
            ax[i].imshow(data[i])
            ax[i].set_title(f"True label: {list(CLASS_IDS.keys())[list(CLASS_IDS.values()).index(tlabels[i])]} \n False label: {list(CLASS_IDS.keys())[list(CLASS_IDS.values()).index(flabels[i])]}")
        plt.show()
    # data_vis(train_images, train_labels, 2)
    # data_vis(val_images, val_labels, 2)
    

    ## Problem 1b: Data Statistics
    # TODO: plot/show image stats
    train_data_shape = train_images.shape
    val_data_shape = val_images.shape
    #print(val_images[0])
    #print(train_data_shape)
    #print(val_data_shape)
    train_max = np.max(train_images)
    train_min = np.min(train_images)
    train_mean_c0 = np.mean(train_images[:,0,:,:])
    train_mean_c1 = np.mean(train_images[:,1,:,:])
    train_mean_c2 = np.mean(train_images[:,2,:,:])
    train_std_c0 = np.std(train_images[:,0,:,:])
    train_std_c1 = np.std(train_images[:,1,:,:])
    train_std_c2 = np.std(train_images[:,2,:,:])
    val_max = np.max(val_images)
    val_min = np.min(val_images)
    val_mean_c0 = np.mean(val_images[:,0,:,:])
    val_mean_c1 = np.mean(val_images[:,1,:,:])
    val_mean_c2 = np.mean(val_images[:,2,:,:])
    val_std_c0 = np.std(val_images[:,0,:,:])
    val_std_c1 = np.std(val_images[:,1,:,:])
    val_std_c2 = np.std(val_images[:,2,:,:])
    #print(train_max, train_min, train_mean_c0, train_mean_c1, train_mean_c2, train_std_c0, train_std_c1, train_std_c2)
    #print(val_max, val_min, val_mean_c0, val_mean_c1, val_mean_c2, val_std_c0, val_std_c1, val_std_c2)
    
    

    # preprocessing imagenet data for training (don't change this)
    train_images, train_labels, val_images, val_labels = prep_imagenet_data(train_images, train_labels, val_images, val_labels)

    #new_train_images, new_train_labels = image_augmentation(train_images, train_labels)
 

    ## Problem 2a: Train ConvNet
    # np.random.seed(47)
    # model1 = ConvNet(dropout_probability=0)
    # train_loss, val_loss, train_accuracy, val_accuracy = train(model1, train_images, train_labels, val_images, val_labels)
    #plot(train_loss, val_loss, train_accuracy, val_accuracy)
    
    # Load the model from the checkpoint file
    # with open('best_model.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    # val_l, val_accu, val_y_hat = loaded_model.forward(val_images, val_labels, train=False)
    # print(val_accu)
    

    ## Problem 2b: Train ConvNetThree
    # model2 = ConvNetThree(dropout_probability=0.1)
    # train_loss, val_loss, train_accuracy, val_accuracy = train(model2, train_images, train_labels, val_images, val_labels)
    # plot(train_loss, val_loss, train_accuracy, val_accuracy)
    with open('model_checkpoint_2b.pkl', 'rb') as file:
        loaded_model_2b = pickle.load(file)
    #_, val_accu, val_y_hat = loaded_model_2b.forward(val_images, val_labels, train=False)
    #train_l, train_accu, train_y_hat = loaded_model.forward(train_images, train_labels, train=False) 
    #print("Train Loss: ", train_l, "Train accuracy: ", train_accu)
    #print("Val Loss: ", val_l, "Val accuracy: ", val_accu)
    ## Problem 2c: Train your best model
    # np.random.seed(47)
    # model3 = ConvNetTwo()
    # train_loss, val_loss, train_accuracy, val_accuracy = train(model3, new_train_images, new_train_labels, val_images, val_labels)
    # plot(train_loss, val_loss, train_accuracy, val_accuracy)
    with open('model_checkpoint_2c.pkl', 'rb') as file:
        loaded_model_2c = pickle.load(file)
    #_, val_accu, val_y_hat = loaded_model_2c.forward(val_images, val_labels, train=False)
    # print(val_accu)
    
    
    
    ## Problem 3a: Evaluation
    # TODO: plot confusion matrix and misclassified images on imagenet data
    # _, val_accu_2b, val_y_hat_2b = loaded_model_2b.forward(val_images, val_labels, train=False)
    # _, val_accu_2c, val_y_hat_2c = loaded_model_2c.forward(val_images, val_labels, train=False)
    # true_labels = np.argmax(val_labels, axis=1)
    # print(val_labels.shape)
    # print(true_labels)
    # cm_2b = confusion_matrix(true_labels, val_y_hat_2b)
    # cm_2c = confusion_matrix(true_labels, val_y_hat_2c)
    
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_2b, annot=True, fmt="d", cmap="Blues", xticklabels=[x for x in range(0, 10)], yticklabels=[x for x in range(0, 10)])
    # plt.title("Confusion Matrix for model 2b")
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_2c, annot=True, fmt="d", cmap="Blues", xticklabels=[x for x in range(0, 10)], yticklabels=[x for x in range(0, 10)])
    # plt.title("Confusion Matrix for model 2c")
    # plt.show()
    # np.random.seed(47)
    # misclassified_images_2b = val_y_hat_2b != true_labels
    # misclassified_images_2c = val_y_hat_2c != true_labels
    # misclassified_index_2b = np.where(misclassified_images_2b == True)[0]
    # misclassified_index_2c = np.where(misclassified_images_2c == True)[0]
    # random_five_2b = np.random.choice(misclassified_index_2b, 5)
    # random_five_2c = np.random.choice(misclassified_index_2c, 5)
    # wrong_images_2b = val_images_copy[random_five_2b]
    # wrong_images_2c = val_images_copy[random_five_2c]
    # wrong_labels_2b = val_y_hat_2b[random_five_2b]
    # wrong_labels_2c = val_y_hat_2c[random_five_2c]
    # correct_labels_2b = true_labels[random_five_2b]
    # correct_labels_2c = true_labels[random_five_2c]
    #data_vis(wrong_images_2b, correct_labels_2b, wrong_labels_2b, 5)
    #data_vis(wrong_images_2c, correct_labels_2c, wrong_labels_2c, 5)
    ## Problem 3b: Evaluate on COCO  "10417-coco.pkl"
    # TODO: Load COCO Data
    with open("10417-coco.pkl", "rb") as f:
        coco_dict = pickle.load(f)
        coco_images = coco_dict["images"]
        coco_images_copy = coco_images
        coco_labels = coco_dict["labels"]
    
    print(CLASS_IDS.keys())

    # TODO: plot COCO data
    #data_vis(coco_images, coco_labels, 2)

    factor = np.max(coco_images)/2
   
    # TODO: get/plot stats COCO
    def prep_imagenet_data_coco(train_images, train_labels):
        # onehot encode labels
        train_labels = one_hot_encode(train_labels)

        # standardize to [-1, 1]
        train_images = (train_images - factor) / factor

        # put channels first
        train_images = np.transpose(train_images, (0, 3, 1, 2))

        return train_images, train_labels

    # TODO: preprocess COCO data, standardize, onehot encode, put channels first
    coco_images, coco_labels = prep_imagenet_data_coco(coco_images, coco_labels)
    _, val_accu_2b, val_y_hat_2b = loaded_model_2b.forward(coco_images, coco_labels, train=False)
    _, val_accu_2c, val_y_hat_2c = loaded_model_2c.forward(coco_images, coco_labels, train=False)
    print(val_accu_2b)
    print(val_accu_2c)
    true_labels = np.argmax(coco_labels, axis=1)
    cm_2b = confusion_matrix(true_labels, val_y_hat_2b)
    cm_2c = confusion_matrix(true_labels, val_y_hat_2c)
    
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_2b, annot=True, fmt="d", cmap="Blues", xticklabels=[x for x in range(0, 10)], yticklabels=[x for x in range(0, 10)])
    # plt.title("Confusion Matrix for model 2b")
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_2c, annot=True, fmt="d", cmap="Blues", xticklabels=[x for x in range(0, 10)], yticklabels=[x for x in range(0, 10)])
    # plt.title("Confusion Matrix for model 2c")
    # plt.show()
    # np.random.seed(47)
    # misclassified_images_2b = val_y_hat_2b != true_labels
    # misclassified_images_2c = val_y_hat_2c != true_labels
    # misclassified_index_2b = np.where(misclassified_images_2b == True)[0]
    # misclassified_index_2c = np.where(misclassified_images_2c == True)[0]
    # random_five_2b = np.random.choice(misclassified_index_2b, 5)
    # random_five_2c = np.random.choice(misclassified_index_2c, 5)
    # wrong_images_2b = val_images_copy[random_five_2b]
    # wrong_images_2c = val_images_copy[random_five_2c]
    # wrong_labels_2b = val_y_hat_2b[random_five_2b]
    # wrong_labels_2c = val_y_hat_2c[random_five_2c]
    # correct_labels_2b = true_labels[random_five_2b]
    # correct_labels_2c = true_labels[random_five_2c]
    # data_vis(wrong_images_2b, correct_labels_2b, wrong_labels_2b, 5)
    # data_vis(wrong_images_2c, correct_labels_2c, wrong_labels_2c, 5)
    
    # np.random.seed(47)
    # misclassified_images_2b = val_y_hat_2b != true_labels
    # misclassified_images_2c = val_y_hat_2c != true_labels
    # misclassified_index_2b = np.where(misclassified_images_2b == True)[0]
    # misclassified_index_2c = np.where(misclassified_images_2c == True)[0]
    # random_five_2b = np.random.choice(misclassified_index_2b, 5)
    # random_five_2c = np.random.choice(misclassified_index_2c, 5)
    # wrong_images_2b = coco_images_copy[random_five_2b]
    # wrong_images_2c = coco_images_copy[random_five_2c]
    # wrong_labels_2b = val_y_hat_2b[random_five_2b]
    # wrong_labels_2c = val_y_hat_2c[random_five_2c]
    # correct_labels_2b = true_labels[random_five_2b]
    # correct_labels_2c = true_labels[random_five_2c]
    # data_vis(wrong_images_2b, correct_labels_2b, wrong_labels_2b, 5)
    # data_vis(wrong_images_2c, correct_labels_2c, wrong_labels_2c, 5)
    # hint: see see prep_imagenet_data() for reference (make sure data range is [-1, 1] before eval!)
    
    # TODO: get loss and accuracy COCO

    # TODO: get confusion matrix COCO and misclassified images COCO
