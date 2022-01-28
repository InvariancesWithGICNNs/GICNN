import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import multivariate_normal
from typing import Tuple, List
from scipy import optimize
from sklearn.cluster import KMeans

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

tfb = tfp.bijectors
tfd = tfp.distributions

tf.config.run_functions_eagerly(False)


def create_bijections(num_layers=3, hidden_units=128, ndim=2) -> List[tfb.Bijector]:
    """
    Return list of bijection layers
    """
    my_bijects = []
    # loop over desired bijectors and put into list
    for i in range(num_layers):
        # Syntax to make a MAF
        anet = tfb.AutoregressiveNetwork(
            params=ndim, hidden_units=[hidden_units, hidden_units],
            activation='elu'
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)
        # Add bijector to list
        my_bijects.append(ab)
        # Now permuate (!important!)
        permute = tfb.Permute([1, 0])
        my_bijects.append(permute)
    # return list of bijectors
    return my_bijects


class PositiveDense(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, units=32, activation="elu"):
        super().__init__()
        self.include_bias = use_bias
        self.units = units
        self.activation = layers.Activation(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, self.units), dtype="float32"),
            trainable=True,
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=self.include_bias
        )

    def call(self, inputs, **kwargs):
        if self.activation is not None:
            return self.activation(tf.matmul(inputs, tf.math.pow(self.w, 2)) + self.b)
        else:
            return tf.matmul(inputs, tf.math.pow(self.w, 2)) + self.b


# class FullyInputConvexBlock(tf.keras.Model):
#     """
#     Equation (2) of input-convexity paper (Fully input convex neural networks)
#     https://arxiv.org/pdf/1609.07152.pdf
#
#     z_i+1 = g_i ( (Wz_i @ z_i + b_i) + Wx_i @ x )
#     """
#
#     def __init__(self, num_layers=3, units=20, activation=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if activation is None:
#             self.g = layers.Activation(activation)
#         else:
#             self.g = StrictlyConvexELU()
#
#         self.Wzs = []
#         self.Wxs = []
#         self.num_layers = num_layers
#
#         self.units = units
#
#     def build(self, input_shape):
#         for i in range(self.num_layers):
#             self.Wxs.append(layers.Dense(units=self.units,
#                                          use_bias=False))
#             self.Wzs.append(PositiveDense(units=self.units,
#                                           use_bias=True))
#
#             self.Wxs[-1].build(input_shape=input_shape)
#             self.Wzs[-1].build(input_shape=input_shape)
#
#     def call(self, inputs, training=None, mask=None):
#         z = inputs
#         for i in range(self.num_layers):
#             z = self.g(self.Wzs[i](z) + self.Wxs[i](inputs))
#         return z


class FullyInputConvexBlock(tf.keras.Model):
    """
    Equation (2) of input-convexity paper (Fully input convex neural networks)
    https://arxiv.org/pdf/1609.07152.pdf

    z_i+1 = g_i ( (Wz_i @ z_i + b_i) + Wx_i @ x )
    """

    def __init__(self, num_layers=3, units=20, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if activation is None:
            self.g = layers.Activation(activation)
        else:
            self.g = StrictlyConvexELU()

        self.Wzs = []
        self.Wxs = []
        self.Wzo = PositiveDense(units=1, use_bias=True)
        self.Wxo = layers.Dense(units=1, use_bias=False)
        self.num_layers = num_layers

        self.units = units

    def build(self, input_shape):
        for i in range(self.num_layers):
            self.Wxs.append(layers.Dense(units=self.units, use_bias=False))
            self.Wzs.append(PositiveDense(units=self.units, use_bias=True))

            self.Wxs[-1].build(input_shape=input_shape)
            self.Wzs[-1].build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        z = inputs
        for i in range(self.num_layers):
            z = self.g(self.Wzs[i](z) + self.Wxs[i](inputs))

        z = self.Wzo(z)

        return z


# class FullyInputConvexBlock_v(tf.keras.Model):
#
#     def __init__(self, units, num_layers, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.units = units
#         self.num_layers = num_layers
#         self.activation = StrictlyConvexELU()
#
#         self.lx1 = Dense_v(units = self.units)
#         self.lx2 = Dense_v(units = self.units)
#         self.lx3 = Dense_v(units = self.units)
#         self.lx4 = Dense_v(units = self.units)
#         self.lx_out = Dense_v(units = 1)
#
#         self.lz1 = PositiveDense_v(units = self.units)
#         self.lz2 = PositiveDense_v(units = self.units)
#         self.lz3 = PositiveDense_v(units = self.units)
#         self.lz_out = PositiveDense_v(units = 1)
#
#     def build(self, input_shape):
#
#         self.lx1.build(input_shape = input_shape)
#         self.lx2.build(input_shape = input_shape)
#         self.lx3.build(input_shape = input_shape)
#         self.lx4.build(input_shape = input_shape)
#         self.lx_out.build(input_shape = input_shape)
#
#         self.lz1.build(input_shape = input_shape)
#         self.lz2.build(input_shape = input_shape)
#         self.lz3.build(input_shape = input_shape)
#         self.lz_out.build(input_shape = input_shape)
#
#     def call(self, inputs, training=None, mask=None):
#         x = inputs
#
#         z1 = self.activation(self.lx1(x))
#         z2 = self.activation(self.lx2(x) + self.lz1(z1))
#         z3 = self.activation(self.lx3(x) + self.lz2(z2))
#         z4 = self.activation(self.lx4(x) + self.lz3(z3))
#         z_out = self.lx_out(x) + self.lz_out(z4)
#
#         return z_out


class FullyInputConvexBlock_v(tf.keras.Model):

    def __init__(self, units, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.units = units
        self.num_layers = num_layers
        self.activation = StrictlyConvexELU()

        self.lx = []
        self.lz = []

        self.lx_out = Dense_v(units = 1)
        self.lz_out = PositiveDense_v(units = 1)

    def build(self, input_shape):

        for i in range(self.num_layers):
            self.lx.append(Dense_v(units = self.units))
            self.lx[-1].build(input_shape = input_shape)

        for i in range(self.num_layers - 1):
            self.lz.append(PositiveDense_v(units = self.units))
            self.lz[-1].build(input_shape = input_shape)

        self.lx_out.build(input_shape = input_shape)
        self.lz_out.build(input_shape = input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        z = self.activation(self.lx[0](x))

        for i in range(self.num_layers - 1):
            z = self.activation(self.lx[i + 1](x) + self.lz[i](z))

        z_out = self.lx_out(x) + self.lz_out(z)

        return z_out


class Dense_v(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_normal_initializer(0.0, 0.1)
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, self.units), dtype="float32"), trainable = True)

        b_init = tf.random_normal_initializer(0.0, 0.1)
        self.b = tf.Variable(initial_value = b_init(shape = (self.units,), dtype = "float32"), trainable = True)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class PositiveDense_v(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        w_init = tf.random_uniform_initializer(-5e-3, 5e-3)
        self.w = tf.Variable(initial_value = w_init(shape=(input_dim, self.units), dtype="float32"), trainable = True)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, tf.math.pow(self.w, 2))


class StrictlyConvexELU(Layer):
    """Variation of ELU.

    It follows:

    ```
      f(x) =  alpha * (exp(x) - 1.) + x for x < 0
      f(x) = x for x >= 0
    ```

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      alpha: Scale for the negative factor.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(StrictlyConvexELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.elu(inputs, self.alpha) + 1e-5 * inputs

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(StrictlyConvexELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class BijectiveDVIB(tf.keras.Model):
    def __init__(self, input_dim, num_bij_layers=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_bij_layers = num_bij_layers

        self.bijection = tfb.Chain(create_bijections(num_bij_layers, 128, ndim=input_dim))

        # little hack for initialization of weights
        self.bijection.inverse(np.ones(input_dim, dtype="float32"))

        # self.logsd_qz = tf.keras.Sequential([
        #     layers.InputLayer((self.input_dim,)),
        #     layers.Dense(units=128, activation="relu"),
        #     layers.Dense(units=int(self.input_dim * (self.input_dim + 1) / 2), ),
        # ])

        self.decoder_y = tf.keras.Sequential(
            [
                layers.InputLayer((self.input_dim,)),
                # layers.Layer()
                # keras.Sequential([BijectiveDense(activation="elu") for i in range(10)]),
                FullyInputConvexBlock_v(units=512, num_layers=4)
                # FullyInputConvexBlock(num_layers=4, units=512, activation="StrictlyConvexELU"),
                # FullyInputConvexBlock(num_layers=1, units=1, activation=None),
                # layers.Dense(units=256, activation="relu"),
                # layers.Dense(units=128, activation="relu"),
                # layers.Dense(units=128, activation="relu"),
                # layers.Dense(units=1),
                # FullyInputConvexBlock(num_layers=int(np.prod(label_shape)), units=1, activation=None),
                # PositiveDense(units=1, activation=None),
            ], name="Decoder_y"
        )

        self.prior_z = tfd.MultivariateNormalDiag(tf.zeros(self.input_dim),
                                                  tf.ones(self.input_dim))
        # self.logsd_qz = tf.Variable(-0.001, trainable=True)
        self.logsd_px = tf.Variable(tf.math.log(0.23), trainable=True)
        self.logsd_py = tf.Variable(tf.math.log(0.3), trainable=True)
        self.logsd_qz = tf.Variable(tf.math.log(.05), trainable=True)
        #self.logsd_qz = tf.Variable(tf.math.log(.05) * tf.ones(2), trainable = True)

        # self.kl_lambda = .1
        starting_lambda = 20.
        self.kl_lambda = tf.Variable(starting_lambda, trainable=False)

        # issue: https://github.com/tensorflow/probability/issues/355, https://github.com/tensorflow/probability/issues/946
        # need to add bijector's trainable variables as an attribute (name does not matter)
        # otherwise this layer has zero trainable variables
        self._variables1 = self.bijection.variables  # https://github.com/tensorflow/probability/issues/355
        self._variables2 = self.bijection.variables  # https://github.com/tensorflow/probability/issues/355

    def q_z(self, x: tf.Tensor) -> tfd.MultivariateNormalDiag:
        """
        Encoder (variational approx q(z|x) to posterior in VAE)
        :param x:
        :return: encoded_posterior
        """
        mu_z = self.bijection.inverse(x)
        sd_z = tf.math.exp(self.logsd_qz)
        encoded_posterior = tfd.MultivariateNormalDiag(mu_z, scale_identity_multiplier=sd_z)
        #encoded_posterior = tfd.MultivariateNormalDiag(mu_z, scale_diag = sd_z)

        return encoded_posterior

    def p_x(self, z: tf.Tensor):
        mu_x_hat = self.bijection.forward(z)
        pdf_x_hat = tfd.Independent(tfd.Normal(mu_x_hat, tf.math.exp(self.logsd_px)),
                                    reinterpreted_batch_ndims=1)
        return pdf_x_hat

    def p_y(self, z: tf.Tensor):
        mu_y_hat = self.decoder_y(z)
        pdf_y_hat = tfd.Independent(tfd.Normal(mu_y_hat, tf.math.exp(self.logsd_py)),
                                    reinterpreted_batch_ndims=1)

        return pdf_y_hat

    def calc_losses(self, x, y):
        z_posterior = self.q_z(x)
        z_sample = z_posterior.sample()
        pdf_x_hat = self.p_x(z=z_sample)
        pdf_y_hat = self.p_y(z=z_sample)

        # compare the shape of x and y with the x_hat and y_hat shapes to avoid broadcasting errors
        tf.debugging.assert_shapes([(x, ('N', 'x')),
                                    (pdf_x_hat.mean(), ('N', 'x')),
                                    (y, ('N', 'y')),
                                    (pdf_y_hat.mean(), ('N', 'y'))
                                    ])

        logprob_x = pdf_x_hat.log_prob(x)
        logprob_y = pdf_y_hat.log_prob(y)
        kl_div_mb = tf.squeeze(z_posterior.kl_divergence(self.prior_z))

        # make sure that the distributions aggregate over the correct dimensions
        tf.debugging.assert_shapes([(logprob_x, ('N',)),
                                    (logprob_y, ('N',)),
                                    (kl_div_mb, ('N',))
                                    ])
        kl_div = tf.reduce_mean(kl_div_mb)  # [mb_size] -> ()
        exp_ll_x = tf.reduce_mean(logprob_x)  # [mb_size] -> ()
        exp_ll_y = tf.reduce_mean(logprob_y)  # [mb_size] -> ()

        elbo = (exp_ll_x + 1.0 * exp_ll_y) - self.kl_lambda * kl_div
        # elbo = -self.kl_lambda * kl_div

        return elbo, pdf_y_hat.mean()

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        One optimization step.
        Calculates losses, performs gradient descent step & logs metrics.
        :param data:
        :return:
        """
        x, y = data

        with tf.GradientTape() as tape:
            elbo, y_pred = self.calc_losses(x, y)
            loss = - elbo

        # the actual prediction for our target
        # y_pred = pdf_y_hat.mean()

        # update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # track losses
        self.compiled_metrics.update_state(y, y_pred)  # TODO
        loss_dict = dict(elbo=elbo, std_z=tf.exp(self.logsd_qz),
                         std_x=tf.exp(self.logsd_px), std_y=tf.exp(self.logsd_py),
                         kl_lambda=self.kl_lambda)
        loss_dict.update({m.name: m.result() for m in self.metrics})
        return loss_dict


class LambdaCallback(keras.callbacks.Callback):
    """
    Callback for decreasing the lambda (kl-weight) after a few epochs.
    """

    def __init__(self, decrease_lambda_each: int = 5, lambda_factor=0.9):
        """
            Callback class for lambda decrease.
        Args:
            decrease_lambda_each: Epochs after which lambda is decreased.
            May want to change that to minibatches for big datasets.
            lambda_factor:
        """
        super().__init__()
        self.decrease_lambda_each = decrease_lambda_each
        self.lambda_factor = lambda_factor

    def on_train_batch_end(self, batch, logs=None):
        self.model: BijectiveDVIB
        if batch % self.decrease_lambda_each == 0:
            self.model.kl_lambda.assign(self.model.kl_lambda * self.lambda_factor)

    # def on_epoch_end(self, epoch, logs=None):
    #     self.model: DVIB
    #     if epoch % self.decrease_lambda_each == 0:
    #         self.model.kl_lambda.assign(self.model.kl_lambda * self.lambda_factor)


class PrintCallback(keras.callbacks.Callback):
    """
    Callback for decreasing the lambda (kl-weight) after a few epochs.
    """

    def __init__(self, save_plot_each: int = 2):
        """
            Callback class for lambda decrease.
        Args:
            decrease_lambda_each: Epochs after which lambda is decreased.
            May want to change that to minibatches for big datasets.
            lambda_factor:
        """
        super().__init__()
        self.save_plot_each = save_plot_each

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_plot_each == 0:
            qz_train = ib.q_z(X_train)
            y_hat_train = ib.decoder_y(qz_train.mean()).numpy()
            X_hat_train = ib.bijection.forward(qz_train.mean()).numpy()

            qz_test = ib.q_z(X_test)
            y_hat_test = ib.decoder_y(qz_test.mean()).numpy()
            X_hat_test = ib.bijection.forward(qz_test.mean()).numpy()

            mu_x = qz_test.mean().numpy()

            # Latent dim grid

            mu_y_min = np.array([0, 1])
            l2 = 6
            mu_grid = get_2d_grid(n = 40, start = -l2, stop = l2).astype('float32')

            y_hat_grid = ib.decoder_y(mu_grid).numpy()
            X_hat_grid = ib.bijection.forward(mu_grid).numpy()

            # Plot

            subplot_title_size = 6
            subplot_title_fontweight = 'bold'

            tick_size = 6
            matplotlib.rc('xtick', labelsize = tick_size)
            matplotlib.rc('ytick', labelsize = tick_size)

            color_grad = 0.5

            fig, axs = plt.subplots(2, 3)
            fig.tight_layout()

            axs[0, 0].set_title('Gaussian Mixture', fontweight = subplot_title_fontweight, size = subplot_title_size)  # Rosenbrock Function
            axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c = np.around(-Y_train.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
            axs[0, 0].set_aspect('equal', adjustable = 'box')

            l1 = 6
            axs[0, 1].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
            axs[0, 1].scatter(mu_x[:, 0], mu_x[:, 1], c = np.around(-y_hat_train.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
            axs[0, 1].set_aspect('equal', adjustable = 'box')
            axs[0, 1].set_xlim([-l1, l1])
            axs[0, 1].set_ylim([-l1, l1])

            axs[0, 2].set_title('Prediction of X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
            axs[0, 2].scatter(X_hat_test[:, 0], X_hat_test[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
            axs[0, 2].set_aspect('equal', adjustable = 'box')

            axs[1, 1].set_title('Latent Grid', fontweight = subplot_title_fontweight, size = subplot_title_size)

            csx, csy, csz = get_contour_input(mu_grid, y_hat_grid)
            cs = axs[1, 1].contour(csx, csy, csz, 6, colors = 'k')
            axs[1, 1].clabel(cs, inline = True, fontsize = 6, fmt = '%1.1f', colors = 'black')

            axs[1, 1].scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 1, alpha = 1.0, cmap = 'viridis')
            axs[1, 1].set_aspect('equal', adjustable = 'box')
            axs[1, 1].set_xlim([-l2, l2])
            axs[1, 1].set_ylim([-l2, l2])

            axs[1, 2].set_title('Grid to X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
            axs[1, 2].scatter(X_hat_grid[:, 0], X_hat_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 1, alpha = 1.0, cmap = 'viridis')
            axs[1, 2].set_aspect('equal', adjustable = 'box')

            plt.savefig(os.path.join(plot_path, "bijective_vae_epoch_{}.pdf".format(epoch)))
            plt.close()


class SaveCallback(keras.callbacks.Callback):
    """
    Callback for decreasing the lambda (kl-weight) after a few epochs.
    """

    def __init__(self, save_plot_each: int = 2):
        """
            Callback class for lambda decrease.
        Args:
            decrease_lambda_each: Epochs after which lambda is decreased.
            May want to change that to minibatches for big datasets.
            lambda_factor:
        """
        super().__init__()
        self.save_plot_each = save_plot_each

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            ib.save('bijective_vae_rosen_epoch_{}.h5py'.format(epoch), True, True)


def plot():

    qz_test = ib.q_z(X_test)
    y_hat_test = ib.decoder_y(qz_test.mean()).numpy()
    X_hat_test = ib.bijection.forward(qz_test.mean()).numpy()

    mu_x = qz_test.mean().numpy()

    # Latent dim grid

    mu_y_min = np.array([0, 1])
    mu_grid_offset = np.std(mu_x) * 1.0
    center = np.mean(mu_y_min)
    l2 = 6.0
    mu_grid = get_2d_grid(n = 40, start = -l2, stop = l2).astype('float32')

    y_hat_grid = ib.decoder_y(mu_grid).numpy()
    X_hat_grid = ib.bijection.forward(mu_grid).numpy()

    # Plot

    subplot_title_size = 6
    subplot_title_fontweight = 'bold'

    tick_size = 6
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    color_grad = 0.5

    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()

    axs[0, 0].set_title('Rosenbrock Function', fontweight = subplot_title_fontweight, size = subplot_title_size)  # Rosenbrock Function
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c = np.around(-Y_train.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 0].set_aspect('equal', adjustable = 'box')

    l1 = 10
    axs[0, 1].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[0, 1].scatter(mu_x[:, 0], mu_x[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 1].set_aspect('equal', adjustable = 'box')
    axs[0, 1].set_xlim([-l1, l1])
    axs[0, 1].set_ylim([-l1, l1])

    axs[0, 2].set_title('Prediction of X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[0, 2].scatter(X_hat_test[:, 0], X_hat_test[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 2].set_aspect('equal', adjustable = 'box')

    axs[1, 0].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 0].scatter(mu_x[:, 0], mu_x[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[1, 0].set_aspect('equal', adjustable = 'box')
    axs[1, 0].set_xlim([-l2, l2])
    axs[1, 0].set_ylim([-l2, l2])

    #csx, csy, csz = get_contour_input(mu_grid, y_hat_grid)
    #cs = axs[1, 1].contour(csx, csy, csz, 6, colors = 'k')
    #axs[1, 1].clabel(cs, inline = True, fontsize = 6, fmt = '%1.1f', colors = 'black')
    axs[1, 1].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 1].set_title('Latent Grid', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 1].scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[1, 1].set_aspect('equal', adjustable = 'box')
    axs[1, 1].set_xlim([-l2, l2])
    axs[1, 1].set_ylim([-l2, l2])

    axs[1, 2].set_title('Grid to X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 2].scatter(X_hat_grid[:, 0], X_hat_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 1, alpha = 1.0, cmap = 'viridis')
    axs[1, 2].set_aspect('equal', adjustable = 'box')
    #axs[1, 2].set_xlim([-0.8, 1])
    #axs[1, 2].set_ylim([-1, 1])

    plt.savefig("plot.pdf")
    plt.close()


def plot_latent_grid_parametrisation():

    def min_search():

        def fu(x):
            y_hat = ib.decoder_y(np.asarray([x])).numpy()
            return y_hat

        x_min = optimize.fmin(fu, np.zeros(2))

        return x_min, fu(x_min)

    def level_set_search(x_init, y_target):

        def fu(x):
            y_hat = ib.decoder_y(np.asarray([x])).numpy()
            return np.mean(np.square(y_target - y_hat))

        x_opt = optimize.fmin(fu, x_init)

        return x_opt

    def level_set_search_polar(center, theta, y_target, fu_tri):

        def fu(x):
            p = np.zeros((1, 2))
            p[:, 0] = x[0]
            p[:, 1:] = theta
            z = center + fu_tri(p)
            y = ib.decoder_y(z).numpy()
            return np.mean(np.square(y_target - y))

        r_opt = optimize.fmin(fu, np.array([1.0]), disp = False)

        return r_opt

    def get_level_set_conture(n_theta, dim, theta_pos, center, y):

        theta = np.zeros((n_theta, dim - 1)) + np.pi / 2
        theta[:, theta_pos] = np.linspace(0, 2 * np.pi, n_theta)

        level_set_conture = np.zeros((n_theta, dim))

        for i, theta_i in enumerate(theta):
            r_opt = level_set_search_polar(center, theta_i, y, spherical_to_cartesian)

            polar_coord_i = np.zeros((1, dim))
            polar_coord_i[:, 0] = r_opt[0]
            polar_coord_i[:, 1:] = theta_i

            coord_i = center + spherical_to_cartesian(polar_coord_i)[0]
            level_set_conture[i, :] = coord_i

        return level_set_conture

    def spherical_to_cartesian(arr):
        r = arr[:, [0]]
        angles = arr[:, 1:]
        sin_prods = np.cumprod(np.sin(angles), axis = 1)
        x1 = r * np.cos(angles[..., :1])
        xs = r * sin_prods[..., :-1] * np.cos(angles[..., 1:])
        xn = r * sin_prods[..., -1:]

        return np.concatenate([x1, xs, xn], -1)

    qz_test = ib.q_z(X_test)
    y_hat_test = ib.decoder_y(qz_test.mean()).numpy()
    X_hat_test = ib.bijection.forward(qz_test.mean()).numpy()

    mu_x = qz_test.mean().numpy()

    # Latent grid

    x_min, y_min = min_search()

    print(y_min)

    center = np.mean(x_min)
    l2 = 8  # 60
    # l2_min = center - l2 - 5  # 10
    # l2_max = center + l2 - 0
    # mu_grid = get_2d_grid(n = 40, start = l2_min, stop = l2_max).astype('float32')

    x_margin = 2.5
    y_margin = 2.5
    start_x = x_min[0] - x_margin - 0.5
    stop_x = x_min[0] + x_margin
    start_y = x_min[1] - y_margin
    stop_y = x_min[1] + y_margin
    #mu_grid = get_2d_grid_2(n = 40, start_x = start_x, stop_x = stop_x, start_y = start_y, stop_y = stop_y).astype('float32')
    mu_grid = get_2d_grid_2(n = 40, start_x = -4, stop_x = 1.5, start_y = -3, stop_y = 2.5).astype('float32')

    y_hat_grid = ib.decoder_y(mu_grid).numpy()
    X_hat_grid = ib.bijection.forward(mu_grid).numpy()

    # Custom level set search

    y_target = 2.0  # 2.45
    level_set_coord = level_set_search(x_min, y_target)

    r = 100.0
    theta = np.linspace(0, 2 * np.pi, 200)
    x = x_min[0] + r * np.cos(theta)
    y = x_min[1] + r * np.sin(theta)
    arc_x = x_min[0] + 0.2 * np.cos(theta)
    arc_y = x_min[1] + 0.2 * np.sin(theta)

    level_set_conture_init = np.concatenate((np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1)), axis = 1)
    level_set_conture = np.zeros_like(level_set_conture_init)

    for i, theta_i in enumerate(theta):
        r_opt = level_set_search_polar(x_min, theta_i, y_target, spherical_to_cartesian)
        coord_i = x_min + spherical_to_cartesian(np.array([[r_opt[0], theta_i]]))[0]
        level_set_conture[i, :] = coord_i

    # Decode level set conture

    y_hat_conture = ib.decoder_y(level_set_conture.astype('float32')).numpy()
    X_hat_conture = ib.bijection.forward(level_set_conture.astype('float32')).numpy()

    # Plot

    subplot_title_size = 6
    subplot_title_fontweight = 'bold'

    tick_size = 6
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    color_grad = 0.25

    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()

    axs[0, 0].set_title('Rosenbrock Function', fontweight = subplot_title_fontweight, size = subplot_title_size)  # Rosenbrock Function
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c = np.around(-Y_train.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 0].set_aspect('equal', adjustable = 'box')

    l1 = 10
    axs[0, 1].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[0, 1].scatter(mu_x[:, 0], mu_x[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 1].set_aspect('equal', adjustable = 'box')
    axs[0, 1].set_xlim([-l1, l1])
    axs[0, 1].set_ylim([-l1, l1])

    axs[0, 2].set_title('Prediction of X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[0, 2].scatter(X_hat_test[:, 0], X_hat_test[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[0, 2].set_aspect('equal', adjustable = 'box')
    axs[0, 2].scatter(X_hat_conture[:, 0], X_hat_conture[:, 1], alpha = 0.8, marker = "o", c = "black", s = 1.0)

    axs[1, 0].set_title('Latent Code of X', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 0].scatter(mu_x[:, 0], mu_x[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[1, 0].set_aspect('equal', adjustable = 'box')
    axs[1, 0].set_xlim([-l2, l2])
    axs[1, 0].set_ylim([-l2, l2])

    ia = 120
    #csx, csy, csz = get_contour_input(mu_grid, y_hat_grid)
    #cs = axs[1, 1].contour(csx, csy, csz, 6, colors = 'k')
    #axs[1, 1].clabel(cs, inline = True, fontsize = 6, fmt = '%1.1f', colors = 'black')
    axs[1, 1].set_title('Latent Grid', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 1].scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 3, alpha = 1.0, cmap = 'viridis')
    axs[1, 1].set_aspect('equal', adjustable = 'box')
    #axs[1, 1].set_xlim([l2_min, l2_max])
    #axs[1, 1].set_ylim([l2_min, l2_max])
    axs[1, 1].plot(level_set_conture[:, 0], level_set_conture[:, 1], alpha = 1.0, c = "black", linewidth = 0.5)
    axs[1, 1].plot(level_set_conture[0:ia, 0], level_set_conture[0:ia, 1], alpha = 1.0, c = "black", linewidth = 1)
    axs[1, 1].plot(arc_x[0:ia], arc_y[0:ia], alpha = 1.0, c = "black", linewidth = 1)
    axs[1, 1].plot([x_min[0], level_set_conture[0, 0]], [x_min[1], level_set_conture[0, 1]], alpha = 1.0, c = "black", linewidth = 1)
    axs[1, 1].plot([x_min[0], level_set_conture[ia - 1, 0]], [x_min[1], level_set_conture[ia - 1, 1]], alpha = 1.0, c = "black", linewidth = 1)
    axs[1, 1].scatter(x_min[0], x_min[1], alpha = 0.8, marker = "o", c = "black", s = 1.0)
    axs[1, 1].scatter(level_set_conture[0, 0], level_set_conture[0, 1], alpha = 0.8, marker = "o", c = "black", s = 1.0)
    axs[1, 1].scatter(level_set_conture[ia - 1, 0], level_set_conture[ia - 1, 1], alpha = 0.8, marker = "o", c = "black", s = 1.0)

    axs[1, 2].set_title('Grid to X and Y', fontweight = subplot_title_fontweight, size = subplot_title_size)
    axs[1, 2].scatter(X_hat_grid[:, 0], X_hat_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 1, alpha = 1.0, cmap = 'viridis')
    axs[1, 2].set_aspect('equal', adjustable = 'box')
    #axs[1, 2].set_xlim([-0.8, 1])
    #axs[1, 2].set_ylim([-1, 1])
    axs[1, 2].set_xlim([-0.4, 0.4])
    axs[1, 2].set_ylim([-0.4, 0.4])

    plt.savefig("plot.pdf")
    plt.close()

    # Individual plot Latent Grid

    subplot_title_size = 18
    tick_size = 14

    plt.rc('figure', figsize=(5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    fig, ax = plt.subplots()

    fig.tight_layout()

    ia = 184  # 137
    color_1 = 'blue'
    line_width_1 = 1.0
    line_width_2 = 4.0
    dot_width = 20.0
    ax.set_title('Bijective Mapping', fontweight = subplot_title_fontweight, size = subplot_title_size)
    ax.scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 30, alpha = 1.0, cmap = 'viridis')

    ax.plot(level_set_conture[:, 0], level_set_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_1)
    ax.plot(level_set_conture[0:ia, 0], level_set_conture[0:ia, 1], alpha = 1.0, c = "black", linewidth = line_width_2)
    ax.plot(arc_x[0:ia], arc_y[0:ia], alpha = 1.0, c = color_1, linewidth = line_width_1)
    ax.plot([x_min[0], level_set_conture[0, 0]], [x_min[1], level_set_conture[0, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    ax.plot([x_min[0], level_set_conture[ia - 1, 0]], [x_min[1], level_set_conture[ia - 1, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)

    ax.scatter(x_min[0], x_min[1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)
    ax.scatter(level_set_conture[0, 0], level_set_conture[0, 1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)
    ax.scatter(level_set_conture[ia - 1, 0], level_set_conture[ia - 1, 1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)

    ax.set_aspect('equal', adjustable = 'box')

    plt.savefig("plot_latent_grid.pdf")
    plt.close()

    # Individual plot level set in input space

    cx = 0.1
    cy = 0.1
    plt.rc('figure', figsize = (5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    fig, ax = plt.subplots()

    fig.tight_layout()

    ax.set_title('Input Space', fontweight = subplot_title_fontweight, size = subplot_title_size)
    ax.scatter(X_hat_test[:, 0], X_hat_test[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 30, alpha = 1.0, cmap = 'viridis')
    ax.plot(X_hat_conture[:, 0], X_hat_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_1)
    ax.plot(X_hat_conture[0:ia, 0], X_hat_conture[0:ia, 1], alpha = 1.0, c = "black", linewidth = line_width_2)

    ax.scatter(cx, cy, alpha = 1.0, marker = "o", c = color_1, s = dot_width)
    ax.scatter(X_hat_conture[0, 0], X_hat_conture[0, 1], alpha = 1.0, marker = "o", c = color_1, s = dot_width)
    ax.scatter(X_hat_conture[ia - 1, 0], X_hat_conture[ia - 1, 1], alpha = 1.0, marker = "o", c = color_1, s = dot_width)

    ax.plot([cx, X_hat_conture[0, 0]], [cy, X_hat_conture[0, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    ax.plot([cx, X_hat_conture[ia - 1, 0]], [cy, X_hat_conture[ia - 1, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)

    margin = 0.04
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_xlim([-0.4 - margin, 0.4 + margin])
    ax.set_ylim([-0.4 - margin, 0.4 + margin])

    plt.savefig("plot_input_space.pdf")
    plt.close()

    # # Individual plot level set in input space - From mu grid
    #
    # plt.rc('figure', figsize = (5, 5))
    # matplotlib.rc('xtick', labelsize = tick_size)
    # matplotlib.rc('ytick', labelsize = tick_size)
    # plt.rcParams["lines.linewidth"] = 3.0
    #
    # fig, ax = plt.subplots()
    #
    # fig.tight_layout()
    #
    # center = np.mean(x_min)
    # # l2 = 10  # 18
    # # l2_min = center - l2 - 0
    # # l2_max = center + l2 - 0
    # # l2_min = -l2
    # # l2_max = l2
    # # nc = 500
    # # mu_grid = get_2d_grid(n = nc, start = l2_min, stop = l2_max).astype('float32')
    #
    # x_margin = 15
    # y_margin = 15
    # start_x = x_min[0] - x_margin - 4
    # stop_x = x_min[0] + x_margin
    # start_y = x_min[1] - y_margin
    # stop_y = x_min[1] + y_margin
    # nc = 500
    # mu_grid = get_2d_grid_2(n = nc, start_x = start_x, stop_x = stop_x, start_y = start_y, stop_y = stop_y).astype('float32')
    #
    # x_margin = 8
    # y_margin = 8
    # start_x = x_min[0] - x_margin - 4
    # stop_x = x_min[0] + x_margin
    # start_y = x_min[1] - y_margin
    # stop_y = x_min[1] + y_margin
    # nc_2 = 100
    # mu_grid_2 = get_2d_grid_2(n = nc_2, start_x = start_x, stop_x = stop_x, start_y = start_y, stop_y = stop_y).astype('float32')
    #
    # #grid_x = get_2d_grid(n = 100, start = -0.8, stop = 0.8).astype('float32')
    #
    # # mu_grid = ib.bijection.inverse(grid_x).numpy()
    # # y_hat_grid_large = ib.decoder_y(mu_grid).numpy()
    # # X_hat_grid_large = ib.bijection.forward(mu_grid).numpy()
    #
    # y_hat_grid_large = ib.decoder_y(mu_grid).numpy()
    # X_hat_grid_large = ib.bijection.forward(mu_grid).numpy()
    #
    # y_hat_grid_large_2 = ib.decoder_y(mu_grid_2).numpy()
    # X_hat_grid_large_2 = ib.bijection.forward(mu_grid_2).numpy()
    #
    # # qidx = np.where((X_hat_grid_large[:, 0] > -0.5) & (X_hat_grid_large[:, 0] < 0.5) & (X_hat_grid_large[:, 1] > -0.2) & (X_hat_grid_large[:, 1] < 0.8))
    # # X_hat_grid_large = X_hat_grid_large[qidx]
    # # y_hat_grid_large = y_hat_grid_large[qidx]
    # # X_hat_grid_large = X_hat_grid_large[:1600]
    # # y_hat_grid_large = y_hat_grid_large[:1600]
    # # #qn = int(np.floor(np.sqrt(len(X_hat_grid_large))))
    # # qn = 40
    #
    # #ax.add_patch(matplotlib.patches.Rectangle((-0.4, -0.4), 0.8, 0.8, fc = 'none', ec = 'black', lw = 2, zorder=10))
    #
    # c_x = 0
    # c_y = 0
    # x_margin = 0.4
    # y_margin = 0.4
    # start_x = c_x - x_margin
    # stop_x = c_x + x_margin
    # start_y = c_y - y_margin
    # stop_y = c_y + y_margin
    # nc_3 = 100
    # x_grid_3 = get_2d_grid_2(n = nc_3, start_x = start_x, stop_x = stop_x, start_y = start_y, stop_y = stop_y).astype('float32')
    #
    # qz_grid = ib.q_z(x_grid_3)
    # y_hat_test = ib.decoder_y(qz_grid.mean()).numpy()
    # X_hat_test = ib.bijection.forward(qz_grid.mean()).numpy()
    #
    # csx, csy, csz = get_contour_input(X_hat_test, y_hat_test.flatten(), n = nc_3)
    # cs = ax.contour(csx, csy, csz, 12, zorder=1)
    # ax.clabel(cs, inline = True, fontsize = 14, fmt = '%1.1f', colors = 'black')
    # ax.set_title('Input Space', fontweight = subplot_title_fontweight, size = subplot_title_size)
    # #ax.scatter(X_hat_grid_large_2[:, 0], X_hat_grid_large_2[:, 1], c = np.around(-y_hat_grid_large_2.flatten() / color_grad, 0), s = 30, alpha = 0.5, cmap = 'viridis')
    # # ax.plot(X_hat_conture[:, 0], X_hat_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_1)
    # # ax.plot(X_hat_conture[0:ia, 0], X_hat_conture[0:ia, 1], alpha = 1.0, c = "black", linewidth = line_width_2)
    # #
    # # ax.scatter(cx, cy, alpha = 1.0, marker = "o", c = color_1, s = dot_width)
    # # ax.scatter(X_hat_conture[0, 0], X_hat_conture[0, 1], alpha = 1.0, marker = "o", c = color_1, s = dot_width)
    # # ax.scatter(X_hat_conture[ia - 1, 0], X_hat_conture[ia - 1, 1], alpha = 1.0, marker = "o", c = color_1, s = dot_width)
    # #
    # # ax.plot([cx, X_hat_conture[0, 0]], [cy, X_hat_conture[0, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    # # ax.plot([cx, X_hat_conture[ia - 1, 0]], [cy, X_hat_conture[ia - 1, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    #
    # ax.set_aspect('equal', adjustable = 'box')
    # # ax.set_xlim([-0.5, 0.5])
    # # ax.set_ylim([-0.2, 0.8])
    # l9 = 1.0  # 1.25
    # ax.set_xlim(l9 * np.array([-0.4, 0.4]))
    # ax.set_ylim(l9 * np.array([-0.4, 0.4]))
    #
    # plt.savefig("plot_input_space_from_grid.pdf")
    # plt.close()
    #
    # # Individual plot Latent Grid large
    #
    # subplot_title_size = 18
    # tick_size = 14
    #
    # plt.rc('figure', figsize = (5, 5))
    # matplotlib.rc('xtick', labelsize = tick_size)
    # matplotlib.rc('ytick', labelsize = tick_size)
    #
    # fig, ax = plt.subplots()
    #
    # fig.tight_layout()
    #
    # color_1 = 'blue'
    # line_width_1 = 1.0
    # line_width_2 = 4.0
    # dot_width = 20.0
    #
    # csx, csy, csz = get_contour_input(qz_grid.mean().numpy(), -y_hat_test.flatten(), n = nc_3)
    # cs = ax.contour(csx, csy, csz, 12)
    # ax.clabel(cs, inline = True, fontsize = 6, fmt = '%1.1f', colors = 'black')
    # ax.set_title('Latent Space', fontweight = subplot_title_fontweight, size = subplot_title_size)
    # # ax.scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid_large.flatten() / color_grad, 0), s = 30, alpha = 0.5, cmap = 'viridis')
    # #
    # # ax.plot(level_set_conture[:, 0], level_set_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_1)
    # # ax.plot(level_set_conture[0:ia, 0], level_set_conture[0:ia, 1], alpha = 1.0, c = "black", linewidth = line_width_2)
    # # ax.plot(arc_x[0:ia], arc_y[0:ia], alpha = 1.0, c = color_1, linewidth = line_width_1)
    # # ax.plot([x_min[0], level_set_conture[0, 0]], [x_min[1], level_set_conture[0, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    # # ax.plot([x_min[0], level_set_conture[ia - 1, 0]], [x_min[1], level_set_conture[ia - 1, 1]], alpha = 1.0, c = color_1, linewidth = line_width_1)
    # #
    # # ax.scatter(x_min[0], x_min[1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)
    # # ax.scatter(level_set_conture[0, 0], level_set_conture[0, 1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)
    # # ax.scatter(level_set_conture[ia - 1, 0], level_set_conture[ia - 1, 1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)
    #
    # #ax.set_xticks([-10, -5, 0, 5, 10])
    # #ax.set_yticks([-10, -5, 0, 5, 10])
    #
    # ax.set_aspect('equal', adjustable = 'box')
    # l9 = 1.2
    # #ax.set_xlim(l9 * np.array([-8, 8]))
    # #ax.set_ylim(l9 * np.array([-6, 10]))
    #
    # plt.savefig("plot_latent_grid_large.pdf")
    # plt.close()

    # ============================
    # Plot multiple custom sampled level sets
    # ============================

    line_width_1 = 2.0

    r = 1.0
    theta = np.linspace(0, 2 * np.pi, 300)
    x = x_min[0] + r * np.cos(theta)
    y = x_min[1] + r * np.sin(theta)

    plt.rc('figure', figsize = (5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)
    plt.rcParams["lines.linewidth"] = 3.0

    fig, ax = plt.subplots()

    fig.tight_layout()

    ax.set_title('Input Space', fontweight = subplot_title_fontweight, size = subplot_title_size)

    vn = 8
    viridis = cm.get_cmap('viridis', vn)

    print(viridis.colors)
    lsc_lst = []

    for j in range(vn):

        y_target = 1.45 + j * 0.75

        level_set_conture_init = np.concatenate((np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1)), axis = 1)
        level_set_conture = np.zeros_like(level_set_conture_init)

        for i, theta_i in enumerate(theta):
            r_opt = level_set_search_polar(x_min, theta_i, y_target, spherical_to_cartesian)
            coord_i = x_min + spherical_to_cartesian(np.array([[r_opt[0], theta_i]]))[0]
            level_set_conture[i, :] = coord_i

        lsc_lst.append(level_set_conture)

        # Decode level set conture

        X_hat_conture = ib.bijection.forward(level_set_conture.astype('float32')).numpy()

        ax.plot(X_hat_conture[:, 0], X_hat_conture[:, 1], alpha = 1.0, c = viridis.colors[vn - 1 - j], linewidth = line_width_1)

    ax.set_aspect('equal', adjustable = 'box')
    l9 = 2.0
    ax.set_xlim(np.array([-0.6, 0.6]))
    ax.set_ylim(np.array([-0.2, 1.0]))

    plt.savefig("plot_input_space_from_grid_2.pdf")
    plt.close()

    # Individual plot Latent Grid

    subplot_title_size = 18
    tick_size = 14

    plt.rc('figure', figsize = (5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    fig, ax = plt.subplots()

    fig.tight_layout()

    ax.set_title('Bijective Mapping', fontweight = subplot_title_fontweight, size = subplot_title_size)

    for j in range(vn):
        level_set_conture = lsc_lst[j]
        ax.plot(level_set_conture[:, 0], level_set_conture[:, 1], alpha = 1.0, c = viridis.colors[vn - 1 - j], linewidth = line_width_1)

    ax.set_aspect('equal', adjustable = 'box')
    flu = 4
    ax.set_xlim(flu * np.array([-3.25, 2.25]))
    ax.set_ylim(flu * np.array([-3, 2.5]))
    ax.set_xticks([-10, -5, 0, 5])
    ax.set_yticks([-10, -5, 0, 5])

    plt.savefig("plot_latent_grid_2.pdf")
    plt.close()

def plot_graphical_abstract():

    def min_search():

        def fu(x):
            y_hat = ib.decoder_y(np.asarray([x])).numpy()
            return y_hat

        x_min = optimize.fmin(fu, np.zeros(2))

        return x_min, fu(x_min)

    def level_set_search(x_init, y_target):

        def fu(x):
            y_hat = ib.decoder_y(np.asarray([x])).numpy()
            return np.mean(np.square(y_target - y_hat))

        x_opt = optimize.fmin(fu, x_init)

        return x_opt

    def level_set_search_polar(center, theta, y_target, fu_tri):

        def fu(x):
            p = np.zeros((1, 2))
            p[:, 0] = x[0]
            p[:, 1:] = theta
            z = center + fu_tri(p)
            y = ib.decoder_y(z).numpy()
            return np.mean(np.square(y_target - y))

        r_opt = optimize.fmin(fu, np.array([1.0]), disp = False)

        return r_opt

    def get_level_set_conture(n_theta, dim, theta_pos, center, y):

        theta = np.zeros((n_theta, dim - 1)) + np.pi / 2
        theta[:, theta_pos] = np.linspace(0, 2 * np.pi, n_theta)

        level_set_conture = np.zeros((n_theta, dim))

        for i, theta_i in enumerate(theta):
            r_opt = level_set_search_polar(center, theta_i, y, spherical_to_cartesian)

            polar_coord_i = np.zeros((1, dim))
            polar_coord_i[:, 0] = r_opt[0]
            polar_coord_i[:, 1:] = theta_i

            coord_i = center + spherical_to_cartesian(polar_coord_i)[0]
            level_set_conture[i, :] = coord_i

        return level_set_conture

    def spherical_to_cartesian(arr):
        r = arr[:, [0]]
        angles = arr[:, 1:]
        sin_prods = np.cumprod(np.sin(angles), axis = 1)
        x1 = r * np.cos(angles[..., :1])
        xs = r * sin_prods[..., :-1] * np.cos(angles[..., 1:])
        xn = r * sin_prods[..., -1:]

        return np.concatenate([x1, xs, xn], -1)

    qz_test = ib.q_z(X_test)
    y_hat_test = ib.decoder_y(qz_test.mean()).numpy()
    X_hat_test = ib.bijection.forward(qz_test.mean()).numpy()

    mu_x = qz_test.mean().numpy()

    # Latent grid

    x_min, y_min = min_search()

    print(y_min)

    center = np.mean(x_min)
    l2 = 7
    l2_min = center - l2 + 0.5
    l2_max = center + l2 + 0.5
    l2_min = -3
    l2_max = 1
    #mu_grid = get_2d_grid(n = 40, start = l2_min, stop = l2_max).astype('float32')

    x_margin = 2.5
    y_margin = 2.5
    start_x = x_min[0] - x_margin - 0.0
    stop_x = x_min[0] + x_margin
    start_y = x_min[1] - y_margin
    stop_y = x_min[1] + y_margin
    #mu_grid = get_2d_grid_2(n = 40, start_x = start_x, stop_x = stop_x, start_y = start_y, stop_y = stop_y).astype('float32')
    mu_grid = get_2d_grid_2(n = 40, start_x = -4, stop_x = 1.5, start_y = -3, stop_y = 2.5).astype('float32')

    y_hat_grid = ib.decoder_y(mu_grid).numpy()
    X_hat_grid = ib.bijection.forward(mu_grid).numpy()

    # Custom level set search

    y_target = 2.0
    level_set_coord = level_set_search(x_min, y_target)

    r = 100.0
    theta = np.linspace(0, 2 * np.pi, 200)
    x = x_min[0] + r * np.cos(theta)
    y = x_min[1] + r * np.sin(theta)
    arc_x = x_min[0] + 0.4 * np.cos(theta)
    arc_y = x_min[1] + 0.4 * np.sin(theta)

    level_set_conture_init = np.concatenate((np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1)), axis = 1)
    level_set_conture = np.zeros_like(level_set_conture_init)

    for i, theta_i in enumerate(theta):
        r_opt = level_set_search_polar(x_min, theta_i, y_target, spherical_to_cartesian)
        coord_i = x_min + spherical_to_cartesian(np.array([[r_opt[0], theta_i]]))[0]
        level_set_conture[i, :] = coord_i

    # Decode level set conture

    y_hat_conture = ib.decoder_y(level_set_conture.astype('float32')).numpy()
    X_hat_conture = ib.bijection.forward(level_set_conture.astype('float32')).numpy()

    # Plot

    subplot_title_size = 18
    tick_size = 14
    subplot_title_fontweight = 'bold'

    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    color_grad = 0.25
    color_1 = 'blue'
    line_width_1 = 4.0
    dot_width = 20.0

    # Individual plot level set in input space

    cx = 0.1
    cy = 0.1
    plt.rc('figure', figsize = (5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    fig, ax = plt.subplots()

    fig.tight_layout()

    ax.set_title('Input Space', fontweight = subplot_title_fontweight, size = subplot_title_size)
    ax.scatter(X_hat_test[:, 0], X_hat_test[:, 1], c = np.around(-y_hat_test.flatten() / color_grad, 0), s = 30, alpha = 1.0, cmap = 'viridis')
    ax.plot(X_hat_conture[:, 0], X_hat_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_1)

    ax.scatter(cx, cy, alpha = 1.0, marker = "o", c = color_1, s = dot_width)

    margin = 0.04
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_xlim([-0.4 - margin, 0.4 + margin])
    ax.set_ylim([-0.4 - margin, 0.4 + margin])

    plt.savefig("plot_graphical_abstract_1.pdf")
    plt.close()

    # Individual plot level set in input space - From mu grid

    plt.rc('figure', figsize = (5, 5))
    matplotlib.rc('xtick', labelsize = tick_size)
    matplotlib.rc('ytick', labelsize = tick_size)

    fig, ax = plt.subplots()

    fig.tight_layout()

    color_1 = 'blue'
    line_width_1 = 1.0
    line_width_2 = 4.0
    dot_width = 20.0
    #csx, csy, csz = get_contour_input(mu_grid, y_hat_grid)
    #cs = ax.contour(csx, csy, csz, 10, colors = 'k')
    #ax.clabel(cs, inline = True, fontsize = 12, fmt = '%1.1f', colors = 'black')
    ax.set_title('Bijective Mapping', fontweight = subplot_title_fontweight, size = subplot_title_size)
    ax.scatter(mu_grid[:, 0], mu_grid[:, 1], c = np.around(-y_hat_grid.flatten() / color_grad, 0), s = 30, alpha = 1.0, cmap = 'viridis')
    ax.plot(level_set_conture[:, 0], level_set_conture[:, 1], alpha = 1.0, c = "black", linewidth = line_width_2)
    ax.scatter(x_min[0], x_min[1], alpha = 0.8, marker = "o", c = color_1, s = dot_width)

    ax.set_aspect('equal', adjustable = 'box')

    plt.savefig("plot_graphical_abstract_2.pdf")
    plt.close()


def get_contour_input(mu_grid, y_grid_pred, n = 40):
    X = mu_grid[:, 0].reshape((n, n))
    Y = mu_grid[:, 1].reshape((n, n))
    Z = y_grid_pred.reshape((n, n))

    return X, Y, Z

def get_contour(mu_grid, y_grid_pred, n = 40):
    X = mu_grid[:, 0].reshape((n, n))
    Y = mu_grid[:, 1].reshape((n, n))
    Z = y_grid_pred.reshape((n, n))

    fig, axs = plt.subplots(3, 3)

    CS = axs[2, 0].contour(X, Y, Z, 6)

    #print(np.array(CS.allsegs[1])[0, :, :])
    #print(CS.allsegs[1][0, :, :])
    #print(len(CS.allsegs))

    # for item in CS.allsegs:
    #     print(item)
    # quit()

    #return CS.allsegs[0][0]  # II-NN Original working
    #return CS.allsegs[1][0]
    return CS

def get_2d_grid(n = None, start = -0.4, stop = 0.4, step = 0.02):
    """Get 2-dimensional grid
    """

    if n is None:
        n = int((stop - start) / step)
        u = np.tile(np.arange(start, stop, step), (n, 1))
        x = np.concatenate((np.reshape(u, (n * n, 1)), np.reshape(u.T, (n * n, 1))), axis = 1)
    else:
        step = (stop - start) / n
        u = np.tile(np.arange(start, stop, step), (n, 1))
        x = np.concatenate((np.reshape(u, (n * n, 1)), np.reshape(u.T, (n * n, 1))), axis = 1)

    return x


def get_2d_grid_2(n = None, start_x = -0.4, stop_x = 0.4, start_y = 0, stop_y = 0):
    """Get 2-dimensional grid
    """

    step_x = (stop_x - start_x) / n
    step_y = (stop_y - start_y) / n
    u = np.tile(np.arange(start_x, stop_x, step_x), (n, 1))
    v = np.tile(np.arange(start_y, stop_y, step_y), (n, 1))
    x = np.concatenate((np.reshape(u, (n * n, 1)), np.reshape(v.T, (n * n, 1))), axis = 1)

    return x


if __name__ == "__main__":
    def gauss_mixture(x):
        mean_1 = [-0.2, 0.0]
        mean_2 = [0.2, 0.0]

        G1 = multivariate_normal(mean_1, [[0.02, 0.0], [0.0, 0.02]])
        G2 = multivariate_normal(mean_2, [[0.02, 0.0], [0.0, 0.02]])

        p = G1.pdf(mean_1) + G2.pdf(mean_1)

        y = -1 * (G1.pdf(x) + G2.pdf(x)) + p
        y = np.expand_dims(y, axis = 1)

        return y.astype("float32")

    def rosen(X, a=1, b=100):
        x1 = X[..., [0]]
        x2 = X[..., [1]]

        y = ((a - 10 * x1) ** 2 + b * (10 * x2 - 100 * x1 ** 2) ** 2) ** 0.25
        return y

    mode = 'eval'
    model_type = 'bijective_rosen'

    if mode == 'train':

        start_time = time.strftime('%m-%d-%Y_%H%M%S')
        ckpt_path = model_type + start_time
        plot_path = os.path.join(ckpt_path, 'plots')
        model_path = os.path.join(ckpt_path, 'models')

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            os.makedirs(plot_path)
            os.makedirs(model_path)

        a = 0.4
        xmin, xmax = -a, a
        ymin, ymax = -a, a
        sqrt_num_test = 40
        num_train = 1600

        X_train = np.stack(np.meshgrid(*(np.linspace(xmin, xmax, sqrt_num_test), np.linspace(ymin, ymax, sqrt_num_test))), -1).astype("float32")
        X_test = np.stack(np.meshgrid(*(np.linspace(xmin, xmax, sqrt_num_test), np.linspace(ymin, ymax, sqrt_num_test))), -1).astype("float32")
        X_train = X_train.reshape(-1, 2)
        X_test = X_test.reshape(-1, 2)

        Y_train, Y_test = rosen(X_train), rosen(X_test)

        n_sqrt_train = 40
        n_sqrt_test = 40
        mb_size = 64

        dataset_train = (tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(n_sqrt_train ** 2).batch(mb_size))
        dataset_test = (tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(n_sqrt_test ** 2).batch(mb_size))

        ib = BijectiveDVIB(input_dim=2, num_bij_layers=4)
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        lambda_cb = LambdaCallback(decrease_lambda_each=30, lambda_factor=0.99)
        print_cb = PrintCallback()
        save_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, 'model_{epoch:02d}_{root_mean_squared_error:.2f}.h5py'),
                                                  monitor='root_mean_squared_error',
                                                  verbose=0,
                                                  save_best_only=True,
                                                    max_to_keep=10)

        ib.compile(optimizer=opt, metrics=[tf.metrics.RootMeanSquaredError()])

        train_hist = ib.fit(x=dataset_train, epochs=50000, verbose=2, callbacks=[lambda_cb, print_cb, save_cb]).history

    elif mode == 'eval':

        #start_time = '01-03-2022_154216'
        #model_name = 'model_1000_0.11.h5py'  # tanh
        #start_time = '01-03-2022_135950'
        #model_name = 'model_486_0.27.h5py'  # elu

        #start_time = '01-04-2022_133957'
        #model_name = 'model_546_0.15.h5py'  # elu  extrapolation strange

        # start_time = '01-04-2022_163234'
        # model_name = 'model_979_0.12.h5py'  # elu

        # start_time = '01-20-2022_180044'
        # model_name = 'model_1770_0.17.h5py'  # softplus 1x Y_loss  5 layers

        start_time = '01-20-2022_163322'
        model_name = 'model_1116_0.20.h5py'  # softplus 1x Y_loss  4 layers model_1474_0.15.h5py  model_2913_0.10.h5py

        start_time = '01-21-2022_124637'
        model_name = 'model_1351_0.10.h5py'  # elu 1x Y_loss  4 layers

        ckpt_path = model_type + start_time
        plot_path = os.path.join(ckpt_path, 'plots')
        model_path = os.path.join(ckpt_path, 'models')

        a = 0.4
        xmin, xmax = -a, a
        ymin, ymax = -a, a
        sqrt_num_test = 40
        num_train = 1600

        X_train = np.stack(np.meshgrid(*(np.linspace(xmin, xmax, sqrt_num_test), np.linspace(ymin, ymax, sqrt_num_test))), -1).astype("float32")
        X_test = np.stack(np.meshgrid(*(np.linspace(xmin, xmax, sqrt_num_test), np.linspace(ymin, ymax, sqrt_num_test))), -1).astype("float32")
        X_train = X_train.reshape(-1, 2)
        X_test = X_test.reshape(-1, 2)

        Y_train, Y_test = rosen(X_train), rosen(X_test)

        ib = BijectiveDVIB(input_dim = 2, num_bij_layers = 4)
        ib.compile(metrics = [tf.metrics.RootMeanSquaredError()])
        ib.load_weights(os.path.join(model_path, model_name))

        #plot_graphical_abstract()
        plot_latent_grid_parametrisation()

