from re import L
import numpy as np
import copy
import pickle
import im2col_helper

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
    print(X_padded)
    
    col_matrix = np.zeros((C*k_height*k_width, out_height*out_width*N))

    for h in range(out_height):
        for w in range(out_width):
            for n in range(N):
                patches = X_padded[n, :, (h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)]
                patches = patches.reshape(C*k_height*k_width, 1)
                col_matrix[:, n + N*(w + out_width*h)] = patches[:,0]
                print(patches[:,0])
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
    X_grad = np.zeros((N, C, H, W))
    out_height = (H - k_height + 2 * padding) // stride + 1
    out_width = (W - k_width + 2 * padding) // stride + 1
    for h in range(out_height):
        for w in range(out_width):
            for n in range(N):
                X_with_padding[n, :, (h*stride):(h*stride+k_height), (w*stride):(w*stride+k_width)] += grad_X_col[:, n + N*(w + out_width*h)].reshape(C, k_height, k_width)
    X_grad = X_with_padding[:, :, padding:(H+padding), padding:(W+padding)]
    return X_grad
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

class Flatten(Transform):
    def forward(self, x):
        """
        Flattens the input tensor.

        Parameters:
        x (ndarray): The input tensor to be flattened.

        Returns:
        ndarray: Flattened input tensor.
        """
        self.input_shape = x.shape  # Store the input shape for later use in backward pass
        return x.reshape(x.shape[0], -1)  # Flatten by preserving batch size

    def backward(self, dloss):
        """
        Computes gradients wrt the input to Flatten.

        Parameters:
        dloss (ndarray): Gradients with respect to the output of Flatten.

        Returns:
        ndarray: Gradients with respect to the input of Flatten.
        """
        return dloss.reshape(self.input_shape)  # Reshape the gradients to match the input shape

# Example usage:
# Initialize the Flatten layer
flatten_layer = Flatten()

example = np.arange(24).reshape((2,3,2,2))
example[1,1,1,1] = -1
example[1,0,0,1] = -1
print(example)
print(np.maximum(0,example))
# Forward pass

# flattened_x = flatten_layer.forward(example)
# print("Flattened input:")
# print(flattened_x)

# # Backward pass

# grads_wrt_input = flatten_layer.backward(flattened_x)
# print("Gradients with respect to input:")
# print(grads_wrt_input)




# example = np.arange(9).reshape((1,1,3,3))
# print(example)
# col_matrix = im2col_helper.im2col(example, 2, 2, 1, 1)
# print(col_matrix)
# backward = im2col_helper.im2col_bw(col_matrix, example.shape, 2, 2, 1, 1)
# print(backward)
# test = im2col_bw(col_matrix, example.shape, 2, 2, 1, 1)
# print(test)