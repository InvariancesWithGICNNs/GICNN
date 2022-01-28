"""
Includes the keras model for the VAE with side-information and cycle consistency loss.
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

from collections import namedtuple
from typing import Tuple
from abc import abstractmethod, ABC

tfd = tfp.distributions


class SideinfoVAE(keras.Model, ABC):
    """
    Basic Abstract Class Implementation of Variational Autoencoder with additional side information Y and cycle consistency loss.
    Requires subclass for defining the used likelihoods of X and Y.

    Consists of encoder, and two decoders.
    One decoder for reconstructing the input X, one for the side information Y.
    """
    Loss = namedtuple("Loss", ['expected_ll_x',
                               'expected_ll_y',
                               'kl_div',
                               'total',
                               "cycle_consistency"])

    def __init__(self, latent_dim: int,
                 encoder_net: layers.Layer,
                 encoder_mu: layers.Layer,
                 encoder_sd: layers.Layer,
                 decoder_x: layers.Layer,
                 decoder_y: layers.Layer,
                 starting_lambda: float = 100.,
                 num_train_samples: int = 1,
                 num_predict_samples: int = 100,
                 num_evaluate_samples: int = 30,
                 *args,
                 **kwargs
                 ) -> None:
        """

        :param latent_dim: Dimension of latent space.
        :param encoder_net: keras layer for the main encoder network.
        :param encoder_mu: keras layer for predicting latent mean. (x_shape)->(latent_dim,)
        :param encoder_sd: keras layer for predicting the latent sd. Must be of shape (x_shape)->(latent_dim,)
        :param decoder_x: keras layer for reconstructing the x from latent dim. (latent_dim,) -> (x_shape)
        :param decoder_y:keras layer for reconstructing the y from latent dim. (latent_dim,) -> (y_shape)
        :param starting_lambda: initial lambda that weighs the KL divergence loss.
        Higher lambda means higher regularization.
        :param num_train_samples: Number of monte carlo samples from encoder for training.
        :param num_predict_samples: Number of monte carlo samples from encoder for predicting.
        :param num_evaluate_samples: Number of monte carlo samples from encoder for evaluating (i.e. during training).
        :param args:
        :param kwargs:
        """
        assert isinstance(starting_lambda, float)
        super().__init__(*args, **kwargs, name="DVIB")

        self.latent_dim = latent_dim

        # Neural Networks for encoder and decoder
        self.encoder_net = encoder_net
        self.encoder_mu = encoder_mu
        self.encoder_sd = encoder_sd
        self.decoder_x = decoder_x
        self.decoder_y = decoder_y

        # The Lagrange parameter which weights the KL Loss.
        # Can and should be slowly decreased during training.
        self.kl_lambda = tf.Variable(starting_lambda, trainable=False)

        # Our custom metrics for the IB
        self.ll_tracker_x = keras.metrics.Mean(name="expected_ll_x")
        self.ll_tracker_y = keras.metrics.Mean(name="expected_ll_y")
        self.kl_tracker = keras.metrics.Mean(name="kl_div")
        self.loss_tracker = keras.metrics.Mean(name="total")

        # how many samples will be used for training/prediction/evaluation..
        self.num_train_samples = num_train_samples
        self.num_predict_samples = num_predict_samples
        self.num_evaluate_samples = num_evaluate_samples

    def prior_z(self) -> tfd.MultivariateNormalDiag:
        """
        Prior of latent variable (p(z)) Usually just standard normal.
        Returned distribution is used for KL divergence
        :return: tfd.MultivariateNormalDiag
        """
        mu = tf.zeros(self.latent_dim)
        rho = tf.ones(self.latent_dim)
        return tfd.MultivariateNormalDiag(mu, rho)

    def q_z(self, x: tf.Tensor, training=None) -> tfd.MultivariateNormalTriL:
        """
        Encoder (variational approx q(z|x) to posterior in VAE)
        :param x:
        :return: encoded_posterior
        """
        mean_branch, sd_branch = tf.split(self.encoder_net(x, training=training), num_or_size_splits=2, axis=-1)

        cov_z = self.encoder_sd(sd_branch)  # + 1e-6  # [mb_size, latent_dim]
        mu_z = self.encoder_mu(mean_branch)  # [mb_size, latent_dim]
        encoded_posterior = tfd.MultivariateNormalTriL(mu_z, tfp.math.fill_triangular(cov_z))

        return encoded_posterior

    def decode(self, z: tf.Tensor) -> Tuple[tfd.Distribution, tfd.Distribution]:
        """
        Decoder.

        Returns the decoded distributions p(x|z), p(y|z).
        :param z: samples in latent space of shape [sample_size, mb_size, 1] which are to be decoded.
        :return: pdf_x_hat, pdf_y_hat
        """
        mu_x_hat = tf.map_fn(self.decoder_x, z)  # [sample_size, mb_size, 1]
        mu_y_hat = tf.map_fn(self.decoder_y, z)  # [sample_size, mb_size, 1]
        pdf_x_hat = self.generate_likelihood_x(mu_x_hat)
        pdf_y_hat = self.generate_likelihood_y(mu_y_hat)
        """
        The expectation is over i) different posterior samples and ii) different data samples in the MB
        Over the output dimension, we want a product over the probability (or sum of logprobs)
        The "batch_shape" tells us which dimensions are averaged over for the expectation
        In our case this is the minibatch size and the number of latent draws in the encoder - each of them
        gives rise to a different distribution.
        The "event_shape" tells us the dimension of the actual distribution event.
        In our case this "event_shape" is the the output dimension, with the dimensions being independent of each other.
        The log probability will be summed over the event shape and averaged over the batch shape.
        
        (if still unclear:
            https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/)
            
        """
        # make Distribution object independent over y shape
        # reinterpreted_batch_ndims says which right-most dims are regarded as the event-size (i.e. the y shape)
        # the remaining are regarded as the 'batch' shape.
        pdf_y_hat = tfd.Independent(pdf_y_hat, reinterpreted_batch_ndims=1)  # event: [2] for 2 sideinfo dims
        pdf_x_hat = tfd.Independent(pdf_x_hat, reinterpreted_batch_ndims=3)  # event: [128, 128, 1]
        # batch_shape=[n_samples, mb_size] event_shape=[output_dim]
        return pdf_x_hat, pdf_y_hat

    def calc_losses(self, x, y, num_samples=1) -> Tuple[Loss, tfd.Distribution]:
        """
        Calculate VAE / DVIB Loss.
        I.e.:
        z_posterior = encoder(x)
        pdf_y_hat = decode(z_posterior.sample())
        loss_vae = -pdf_y_hat.log_prob(y) + lambda * KL(z_posterior, z_prior) = -ELBO

        z_sampled ~ U(-lim, +lim)^[z_dim]
        x_sampled = decode(z_sampled).sample()
        loss_cycle_consistency = decode(encode(x_sampled).sample()).logprob(x_sampled)

        :param x:
        :param y:
        :param num_samples: Number of samples for the reconstruction loss of the VAE, conventionally 1.
        :return: losses, pdf_y_hat
        """
        z_prior = self.prior_z()
        z_posterior = self.q_z(x)
        pdf_x_hat, pdf_y_hat = self.decode(z=z_posterior.sample(num_samples))

        tf.debugging.assert_equal(pdf_y_hat.event_shape, y.shape[1:])
        tf.debugging.assert_equal(pdf_x_hat.event_shape, x.shape[1:])

        # Expected Log Likelihood / Reconstruction error
        logprob_x = pdf_x_hat.log_prob(x)
        logprob_y = pdf_y_hat.log_prob(y)

        cycle_consistency_loss = self.calc_cycle_consistency_loss(pdf_x_hat, z_posterior)

        # KL Divergence
        kl_div_mb = tf.squeeze(z_posterior.kl_divergence(z_prior))

        # make sure that the distributions aggregate over the correct dimensions
        tf.debugging.assert_shapes([(logprob_x, ('S', 'N')),
                                    (logprob_y, ('S', 'N')),
                                    (kl_div_mb, ('N',))
                                    ])
        exp_ll_x = tf.reduce_mean(logprob_x)  # [sample_size, mb_size] -> ()
        exp_ll_y = tf.reduce_mean(logprob_y)  # [sample_size, mb_size] -> ()
        kl_div = tf.reduce_mean(kl_div_mb)  # [sample_size, mb_size] -> ()

        tf.debugging.assert_shapes([(exp_ll_x, ()),
                                    (exp_ll_y, ()),
                                    (kl_div, ())
                                    ])

        # Combined Loss, i.e. negative ELBO
        elbo = .5 * (exp_ll_x + 0.2 * exp_ll_y * 32 ** 2) - self.kl_lambda * kl_div

        additional_losses = 0.01 * cycle_consistency_loss
        total_loss = -elbo + additional_losses
        losses = self.Loss(expected_ll_y=exp_ll_y,
                           expected_ll_x=exp_ll_x,
                           kl_div=kl_div,
                           total=total_loss,
                           cycle_consistency=cycle_consistency_loss
                           )
        return losses, pdf_y_hat

    def calc_cycle_consistency_loss(self, pdf_x_hat, z_posterior):
        # cycle consistency on mini batch X
        z_2 = self.q_z(pdf_x_hat.sample()[0, ...])
        X_2 = self.decoder_x(z_2.sample())
        cycle_cons_train = tf.reduce_mean(pdf_x_hat.log_prob(X_2[tf.newaxis, ...]))
        # cycle consistency on sampled X
        z_uniform_mu = self.sample_z_uniform(z_posterior)
        X_sampled = self.decoder_x(z_uniform_mu)
        z_uniform_mu_cycled = self.q_z(self.decode(z=z_uniform_mu[None, ...])[0].sample()[0, ...]).sample()
        X_cycled_pdf = tfd.Independent(self.generate_likelihood_x(self.decoder_x(z_uniform_mu_cycled)),
                                       reinterpreted_batch_ndims=3)
        cycle_cons_sampled = tf.reduce_mean(X_cycled_pdf.log_prob(X_sampled))
        # combined cycle consistency loss (negative to turn log_prob into loss)
        cycle_consistency_loss = -(cycle_cons_sampled + cycle_cons_train) * 0.5
        return cycle_consistency_loss

    @staticmethod
    def sample_z_uniform(z_posterior):
        """
        Sample z uniformly within an area around the origin.
        The sampled area scales with the absolute value of the (variational) posterior mean,
        so it adapts to different KL divergences (e.g. if the latent codes are further away from the origin).
        :param z_posterior:
        :return:
        """
        mu_var = tf.reduce_mean(z_posterior.mean() ** 2, 0)
        sampling_factor = 2.0
        sampling_limit = sampling_factor * tf.sqrt(mu_var + 1e-4)
        z_uniform_mu = tf.random.uniform(tf.shape(z_posterior.mean()),
                                         -sampling_limit, sampling_limit)
        return z_uniform_mu

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        """
        One optimization step.
        Calculates losses, performs gradient descent step & logs metrics.
        :param data:
        :return:
        """
        x, y, _ = data
        y_transformed = self.target_transform(y)

        with tf.GradientTape() as tape:
            losses, pdf_y_hat = self.calc_losses(x, y_transformed, 1)

        # the actual prediction for our target
        y_pred = self._predict_transform(tf.reduce_mean(pdf_y_hat.mean(), axis=0))

        # update weights
        gradients = tape.gradient(losses.total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # track losses
        self.compiled_metrics.update_state(y, y_pred)  # TODO
        loss_dict = self.update_custom_metrics(losses)
        loss_dict.update({m.name: m.result() for m in self.metrics})

        return loss_dict

    def test_step(self, data):
        """
        Step used by Keras for validation metrics
        :param data:
        :return:
        """
        x, y, _ = data
        y_transformed = self.target_transform(y)

        # calculate losses
        losses, pdf_y_hat = self.calc_losses(x, y_transformed, 30)
        # the actual prediction for our target
        y_pred = self._predict_transform(tf.reduce_mean(pdf_y_hat.mean(), axis=0))

        # track losses
        self.compiled_metrics.update_state(y, y_pred)  # TODO
        loss_dict = self.update_custom_metrics(losses)
        loss_dict.update({m.name: m.result() for m in self.metrics})

        return loss_dict

    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        This gets called when directly calling model(input) and in the predict routine of keras.
        Currently uses the mean of the encoder as latent value and returns the decoded distirbution.

        :param inputs:
        :param training:
        :param mask:
        :return: pdf_y_hat.mean(), pdf_y_hat.stddev()
        """
        x = inputs
        z_posterior = self.q_z(x, training=training)
        # self.decode requires [sample_size, mb_size, 1]
        pdf_x_hat, pdf_y_hat = self.decode(z=z_posterior.mean()[tf.newaxis, ...])
        return pdf_x_hat.mean(), pdf_y_hat.mean()

    def predict_step(self, data):
        """
        This gets called by keras' predict.
        :param data:
        :return:
        """
        x = data
        return self(x, training=False)

    def update_custom_metrics(self, losses: Loss) -> dict:
        """
                Update the all the IB Metrics during training or evaluation.
        :param losses:
        :return:
        """
        self.ll_tracker_x.update_state(losses.expected_ll_x)
        self.ll_tracker_y.update_state(losses.expected_ll_y)
        self.kl_tracker.update_state(losses.kl_div)
        self.loss_tracker.update_state(losses.total)

        loss_dict = losses._asdict()
        loss_dict.update({"lambda": self.kl_lambda})
        return loss_dict

    @abstractmethod
    def generate_likelihood_x(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Set the likelihood of the input reconstruction (x).
        The loc (e.g. the Mean for Normal distribution) is given by the decoder network.
        e.g. return tfd.Normal(loc, self.decoder_sd()) for regression

        :param loc:
        :return:
        """
        raise NotImplementedError("You have to reimplement this..")

    @abstractmethod
    def generate_likelihood_y(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Set the likelihood of the sideinformation (y).
        The loc (e.g. the Mean for Normal distribution) is given by the decoder network.
        e.g. return tfd.Normal(loc, self.decoder_sd()) for regression, binomial for classification

        :param loc:
        :return:
        """
        raise NotImplementedError("You have to reimplement this..")

    def target_transform(self, y: tf.Tensor):
        """
        Could include e.g. onehot transformation for classification.
        Usually just the identity for regression.
        Args:
            y:

        Returns:

        """
        return y

    @staticmethod
    def _predict_transform(mean_prediction):
        """
        Transformation of mean_over_samples(decoder( encoder(x).sample() )) , which is then used for the prediction.
        Usually just the identity, can be argmax for Classification.

        Args:
            mean_prediction:

        Returns:

        """
        return mean_prediction


class RegressionVAE(SideinfoVAE):
    """
    Subclass for DVIB with X as regression target.
    Includes positive transformation for standard_deviation & Normal likelihood.

    """

    def get_config(self):
        pass

    def __init__(self, sideinfo_loglik_min=0.,
                 *args, **kwargs):
        """
        :param sideinfo_loglik_min: 0
        :param latent_dim: Dimension of latent space.
        :param encoder_net: keras layer for the main encoder network.
        :param encoder_mu: keras layer for predicting latent mean. (x_shape)->(latent_dim,)
        :param encoder_sd: keras layer for predicting the latent sd. Must be of shape (x_shape)->(latent_dim,)
        :param decoder_x: keras layer for reconstructing the x from latent dim. (latent_dim,) -> (x_shape)
        :param decoder_y:keras layer for reconstructing the y from latent dim. (latent_dim,) -> (y_shape)
        :param starting_lambda: initial lambda that weighs the KL divergence loss.
        Higher lambda means higher regularization.
        :param num_train_samples: Number of monte carlo samples from encoder for training.
        :param num_predict_samples: Number of monte carlo samples from encoder for predicting.
        :param num_evaluate_samples: Number of monte carlo samples from encoder for evaluating (i.e. during training).

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.sideinfo_loglik_min = sideinfo_loglik_min
        self.decoder_logsd_x = tf.Variable(1.0, trainable=True)
        self.decoder_logsd_y = tf.Variable(1.0, trainable=True)
        assert sideinfo_loglik_min >= 0

    @property
    def decoder_sd_y(self) -> tf.Tensor:
        """
        Wrapper for sd
        Returns:
        """

        return tf.nn.softplus(self.decoder_logsd_y) + self.sideinfo_loglik_min

    @property
    def decoder_sd_x(self) -> tf.Tensor:
        """
        Wrapper for sd
        Returns:
        """

        return tf.nn.softplus(self.decoder_logsd_x)

    def generate_likelihood_x(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Use Gaussian Likelihood with scalar scale.
        Args:
            loc:

        Returns:

        """
        return tfd.Normal(loc, self.decoder_sd_x)

    def generate_likelihood_y(self, loc: tf.Tensor) -> tfd.Distribution:
        """
        Use Gaussian Likelihood with scalar scale.
        Args:
            loc:

        Returns:

        """
        return tfd.Normal(loc, self.decoder_sd_y)


class LambdaCallback(keras.callbacks.Callback):
    """
    Callback for decreasing the lambda (kl-weight) after a few epochs.
    """

    def __init__(self, decrease_lambda_each: int = 5,
                 lambda_factor=0.9, min_lambda=1e-7):
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
        self.min_lambda = min_lambda

    def on_batch_end(self, mb, logs=None):
        """

        :param mb:
        :param logs:
        :return:
        """
        self.model: SideinfoVAE
        if mb % self.decrease_lambda_each == 0 and self.model.kl_lambda > self.min_lambda:
            self.model.kl_lambda.assign(self.model.kl_lambda * self.lambda_factor)

    # def on_epoch_end(self, epoch, logs=None):
    #     self.model: DVIB
    #     if epoch % self.decrease_lambda_each == 0:
    #         self.model.kl_lambda.assign(self.model.kl_lambda * self.lambda_factor)
