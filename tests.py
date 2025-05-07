import unittest
import numpy as np
import cnn
from numpy.testing import assert_allclose
import pickle as pk

seed = 10417617

with open("tests.pk","rb") as f: tests = pk.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestReLU
# to run all tests: python -m unittest tests


# 0pts
class TestReLU(unittest.TestCase):
  def test(self):
    sl = cnn.ReLU()

    test7 = (np.arange(36).reshape((18,2))-18).astype('float64')
    out1 = tests["relu1"]
    out2 = tests["relu2"]
    out3 = tests["relu3"]

    assert_allclose(sl.forward(test7),out1,atol=TOLERANCE)
    assert_allclose(sl.forward(test7),out2,atol=TOLERANCE)

    test8 = np.arange(36).reshape((18,2))-10
    assert_allclose(sl.backward(test8),out3,atol=TOLERANCE)


# 1pts
class TestConvWeightsBias(unittest.TestCase):
  def test(self):
    weights,bias = tests["conv_weights"]
    sl_weights, sl_bias = cnn.Conv((3,32,32),(5,5,5),rand_seed=seed).get_wb_conv()

    assert_allclose(weights,sl_weights,atol=TOLERANCE)
    assert_allclose(bias,sl_bias,atol=TOLERANCE)


# 8pts
class TestIm2Col(unittest.TestCase):
    def test(self):
        im2col_input = tests['im2col_input']
        im2col_output = tests['im2col_output']

        im2col_grad_X_col = tests['im2col_grad_X_col']
        im2col_grad_X = tests['im2col_grad_X']

        assert_allclose(cnn.im2col(im2col_input, 3, 3, 1, 1), im2col_output, atol=TOLERANCE)
        assert_allclose(cnn.im2col_bw(im2col_grad_X_col, im2col_input.shape, 3, 3, 1, 1), im2col_grad_X, atol=TOLERANCE)


# 2pts
class TestConvForward(unittest.TestCase):
    def test(self):
        test_imgs = tests["test_imgs"]
        conv_obj = cnn.Conv(input_shape=(3,32,32),filter_shape=(5,5,5),rand_seed=seed)
        conv_out = conv_obj.forward(test_imgs,1,2)

        assert_allclose(conv_out,tests["conv_out"],atol=TOLERANCE)


# 2pts
class TestConvBackward(unittest.TestCase):
    def test(self):
        dloss = tests["dloss"]
        test_imgs = tests["test_imgs"]
        conv_obj = cnn.Conv(input_shape=(3,32,32),filter_shape=(5,5,5),rand_seed=seed)
        conv_out = conv_obj.forward(test_imgs,1,2)
        conv_back = conv_obj.backward(dloss)

        assert_allclose(conv_back[0],tests["conv_back_w"],atol=TOLERANCE)
        assert_allclose(conv_back[1],tests["conv_back_b"],atol=TOLERANCE)
        assert_allclose(conv_back[2],tests["conv_back_x"],atol=TOLERANCE)


# 1pts
class TestConvUpdate(unittest.TestCase):
    def test(self):
        dloss = tests["dloss"]
        test_imgs = tests["test_imgs"]
        conv_obj = cnn.Conv(input_shape=(3,32,32),filter_shape=(5,5,5),rand_seed=seed)
        conv_out = conv_obj.forward(test_imgs,1,2)
        conv_back = conv_obj.backward(dloss)
        conv_obj.update(learning_rate=0.001,momentum_coeff=0.9)
        sl_weights, sl_bias = conv_obj.get_wb_conv()

        assert_allclose(sl_weights,tests["conv_updated_w"],atol=TOLERANCE)
        assert_allclose(sl_bias,tests["conv_updated_b"],atol=TOLERANCE)


# 3pts
class TestMaxPoolForward(unittest.TestCase):
    def test(self):
        test_imgs = tests["conv_out"]
        max_obj = cnn.MaxPool((2,2),2)
        max_out = max_obj.forward(test_imgs)

        assert_allclose(max_out,tests["max_out"],atol=TOLERANCE)


# 3pts
class TestMaxPoolBackward(unittest.TestCase):
    def test(self):
        dloss = tests["max_pool_dloss"]
        test_imgs = tests["conv_out"]
        max_obj = cnn.MaxPool((2,2),2)
        max_out = max_obj.forward(test_imgs)
        max_back = max_obj.backward(dloss)

        assert_allclose(max_back,tests["max_back"],atol=TOLERANCE)


# 1pts
class TestFCWeightsBias(unittest.TestCase):
    def test(self):
        weights,bias = tests["fc_weights_bias"]
        sl_weights, sl_bias = cnn.LinearLayer(256,20,rand_seed=seed).get_wb_fc()

        assert_allclose(weights,sl_weights,atol=TOLERANCE)
        assert_allclose(bias,sl_bias,atol=TOLERANCE)


# 1pts
class TestLinearForward(unittest.TestCase):
    def test(self):
        features = tests["linear_features"]
        linear_obj = cnn.LinearLayer(256,20,rand_seed=seed)
        linear_out = linear_obj.forward(features)

        assert_allclose(linear_out,tests["linear_out"],atol=TOLERANCE)


# 1pts
class TestLinearBackward(unittest.TestCase):
    def test(self):
        features = tests["linear_features"]
        linear_dloss = tests["linear_dloss"]
        linear_obj = cnn.LinearLayer(256,20,rand_seed=seed)
        linear_out = linear_obj.forward(features)
        linear_grad = linear_obj.backward(linear_dloss)

        assert_allclose(linear_grad[0],tests["linear_grad_w"],atol=TOLERANCE)
        assert_allclose(linear_grad[1],tests["linear_grad_b"],atol=TOLERANCE)
        assert_allclose(linear_grad[2],tests["linear_grad_x"],atol=TOLERANCE)


# 1pts
class TestLinearUpdate(unittest.TestCase):
    def test(self):
        features = tests["linear_features"]
        linear_dloss = tests["linear_dloss"]
        linear_obj = cnn.LinearLayer(256,20,rand_seed=seed)
        linear_out = linear_obj.forward(features)
        linear_grad = linear_obj.backward(linear_dloss)
        linear_obj.update(learning_rate=0.001,momentum_coeff=0.9)
        w,b = linear_obj.get_wb_fc()

        assert_allclose(w,tests["linear_updated_w"],atol=TOLERANCE)
        assert_allclose(b,tests["linear_updated_b"],atol=TOLERANCE)


# 4pts
class TestConvNet(unittest.TestCase):
    def test(self):
        test_imgs = tests['test_imgs']
        labels = tests['true_labels']
        conv = cnn.ConvNet()

        for i in range(10):
            loss, y_pred = conv.forward(test_imgs,labels)
            total_loss = np.sum(loss/labels.shape[0])
            assert_allclose(total_loss,tests["loop_loss"][i],atol=TOLERANCE)
            conv.backward()
            conv.update(0.001,0.9)


