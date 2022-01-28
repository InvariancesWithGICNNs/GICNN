"""
Includes utility keras layers, such as the input convex network layers or convolutional blocks.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers, constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops

from typing import Optional, AnyStr

tfb = tfp.bijectors


class FullyInputConvexBlock2(tf.keras.Model):

    def __init__(self, units, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.units = units
        self.num_layers = num_layers
        self.activation = layers.Activation("softplus")

        self.lx = []
        self.lz = []

        self.lx_out = layers.Dense(units=1)
        self.lz_out = PositiveDense(units=1)

    def build(self, input_shape):

        for i in range(self.num_layers):
            self.lx.append(layers.Dense(units=self.units))
            self.lx[-1].build(input_shape=input_shape)

        for i in range(self.num_layers - 1):
            self.lz.append(PositiveDense(units=self.units, activation=self.activation))
            self.lz[-1].build(input_shape=input_shape)

        self.lx_out.build(input_shape=input_shape)
        self.lz_out.build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        z = self.activation(self.lx[0](x))

        for i in range(self.num_layers - 1):
            z = self.activation(self.lx[i + 1](x) + self.lz[i](z))

        z_out = self.lx_out(x) + self.lz_out(z)

        return z_out


class FullyInputConvexBlock(tf.keras.Model):
    """
    Equation (2) of input-convexity paper (Fully input convex neural networks)
    https://arxiv.org/pdf/1609.07152.pdf

    z_i+1 = g_i ( (Wz_i @ z_i + b_i) + Wx_i @ x )
    """

    def __init__(self, num_layers=3, units=20, activation="softplus", *args, **kwargs):
        """

        :param num_layers:
        :param units:
        :param activation:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if isinstance(activation, str):
            self.g = layers.Activation(activation)
        else:
            self.g = activation()
        self.Wzs = []
        self.Wxs = []
        self.num_layers = num_layers

        self.units = units

    def build(self, input_shape):
        for i in range(self.num_layers):
            self.Wxs.append(layers.Dense(units=self.units,
                                         use_bias=False))
            self.Wzs.append(PositiveDense(units=self.units,
                                          use_bias=True))

            self.Wxs[-1].build(input_shape=input_shape)
            self.Wzs[-1].build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        z = inputs
        for i in range(self.num_layers):
            z = self.g(self.Wzs[i](z) + self.Wxs[i](inputs))
        return z


class PositiveDense(tf.keras.layers.Layer):
    def __init__(self, units=32, activation="relu", use_bias=True):
        super().__init__()
        self.units = units
        if isinstance(activation, str):
            self.activation = layers.Activation(activation)
        else:
            self.activation = activation

        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, self.units), dtype="float32"),
            trainable=True,
        )

        if self.use_bias:
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(
                initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=self.use_bias
            )

    def call(self, inputs, **kwargs):
        out = tf.matmul(inputs, tf.math.pow(self.w, 2))
        if self.use_bias:
            out += self.b

        if self.activation is not None:
            return self.activation(out)
        else:
            return out


class DiagonalPositive(constraints.Constraint):
    """Constrains the weights to be diagonal + positive.
    https://stackoverflow.com/questions/53744518/custom-layer-with-diagonal-weight-matrix/53756678

    """

    def __call__(self, w):
        N = keras.backend.int_shape(w)[-1]
        m = tf.eye(N)
        w = w * m * math_ops.cast(math_ops.greater_equal(w, 0.), keras.backend.floatx())
        return w


def conv2x2(filters, kernel_size=2, strides=1, dilation=1, activation: Optional[AnyStr] = "relu", padding="same"):
    return layers.Conv2D(filters, (kernel_size, kernel_size), strides=strides, padding=padding, activation=activation,
                         dilation_rate=dilation,
                         kernel_initializer=tf.keras.initializers.HeNormal())


class ConvBlock(keras.Model):
    def __init__(self, units, kernel_size, activation="relu",
                 increase_dim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv2x2(units, kernel_size, activation=None)
        self.conv2 = conv2x2(units, kernel_size, activation=None)
        self.add = layers.Add()
        self.out_activation = layers.Activation(activation=activation)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        if increase_dim:
            self.conv_skip = conv2x2(units, 1, activation=None)
        self.increase_dim = increase_dim

    def call(self, inputs, training=None, mask=None):
        x = inputs
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.out_activation(fx)
        fx = self.conv2(fx)
        if self.increase_dim:
            x = self.conv_skip(x)

        out = self.out_activation(self.add([fx, x]))
        return out
