"""
Fashion MNIST Experiment
"""
import os
from typing import Dict
import argparse
import pprint
from pathlib import Path
import tqdm
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from scipy.optimize import minimize_scalar

from lib.VAE import RegressionVAE, LambdaCallback
from lib.keras_layers import FullyInputConvexBlock, conv2x2, ConvBlock, DiagonalPositive
from lib.preprocessing import rotate
from lib.plotting import plot_interpolations, plot_dataset_samples
from lib.spherical_coords import cartesian_to_spherical, spherical_to_cartesian, translate_to_minimum, \
    translate_to_origin, add_radius

tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors
Dataset = tf.data.Dataset

tf.config.run_functions_eagerly(False)

FMNIST_IDS = dict(
    Tshirt=0,
    Trouser=1,
    Pullover=2,
    Dress=3,
    Coat=4,
    Sandal=5,
    Shirt=6,
    Sneaker=7,
    Bag=8,
    Ankleboot=9)


def main():
    # load images + labels, rotate images
    train_, test_ = fashion_mnist.load_data()
    image_shape, label_shape, img_class_shape = [36, 36, 1], (), ()

    target_ids = [FMNIST_IDS[key]
                  for key in ["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Shirt", "Sneaker"]]

    idx_train = np.isin(train_[1], target_ids)
    idx_test = np.isin(test_[1], target_ids)
    train = (train_[0][idx_train], train_[1][idx_train])
    test = (test_[0][idx_test], test_[1][idx_test])

    train_dataset, val_dataset = [
        (Dataset.from_tensor_slices(t_slice).map(rotate, num_parallel_calls=tf.data.AUTOTUNE))
        for t_slice in [train, test]]
    plot_dataset_samples(train_dataset, dir=FILE_DIR / "images")

    train_dataset = train_dataset.cache().shuffle(50000).batch(args.mb_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(args.mb_size).prefetch(tf.data.AUTOTUNE)

    # plot some dataset samples of the rotated FMNIST
    # std_rotation = 1 / 12 * (-np.pi / 4 - np.pi / 4) ** 2
    # print(f"Standard Deviation of Side Information Y: {std_rotation:.2f}")

    # build encoder & decoder networks
    network_layers = build_networks(input_shape=image_shape, label_shape=label_shape)
    model = RegressionVAE(latent_dim=args.latent_dim, starting_lambda=1.,
                          num_train_samples=1,
                          sideinfo_loglik_min=0.01,
                          **network_layers)

    lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=3e-3,
                                                       decay_steps=((train[0].shape[0] // args.mb_size)
                                                                    * args.num_epochs),
                                                       end_learning_rate=5e-5, power=2)
    opt = keras.optimizers.Adamax(learning_rate=lr)

    model.compile(optimizer=opt, metrics=[tf.metrics.RootMeanSquaredError()])

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_expected_ll_x",
                                                      patience=10, mode="max",
                                                      restore_best_weights=True)

    save_cb = tf.keras.callbacks.ModelCheckpoint(filepath=str(FILE_DIR / model_name),
                                                 monitor="val_expected_ll_x",
                                                 max_to_keep=1)

    lambda_cb = LambdaCallback(decrease_lambda_each=20, lambda_factor=0.95)

    if args.train_model:
        try:
            train_history: Dict[str, list] = model.fit(x=train_dataset, epochs=args.num_epochs,
                                                       callbacks=[lambda_cb, early_stopping_cb, save_cb],
                                                       # callbacks=[early_stopping_cb],
                                                       # steps_per_epoch=args.num_epochs//10,
                                                       validation_data=val_dataset,
                                                       verbose=1
                                                       ).history
        except KeyboardInterrupt:
            print("Manually interrupted training..")
            train_history: Dict[str, list] = model.history.history
    else:
        print("Loading model, skipping training..")
        model.load_weights(str(FILE_DIR / model_name))

    # argmin used for the spherical parameterisation
    argmin = find_model_argmin(model)

    # create interpolations between some randomly test images. The selectd test images are then randomly rotated.
    for num, (img1_idx, img2_idx) in tqdm.tqdm(enumerate(np.random.randint(0, test[0].shape[0], size=(10, 2)))):
        img1 = test[0][img1_idx]
        img2 = test[0][img2_idx]

        rotation_angle = np.array([np.random.rand() * np.pi / 2 - np.pi / 4])
        img1_rotated = rotate(img1, None, angle=rotation_angle)[0]
        img2_rotated = rotate(img2, None, angle=rotation_angle)[0]

        img1_rotated_rec, img2_rotated_rec, results = interpolate_images_on_lvlset(argmin,
                                                                                   model,
                                                                                   img1_rotated,
                                                                                   img2_rotated)

        z = tf.concat([translate_to_origin(spherical_to_cartesian(res[1]), argmin) for res in results], 0)
        levelset_images, levelset_y = model.decode(z[:, np.newaxis, :])

        plot_interpolations(levelset_images, levelset_y, img1_rotated, img2_rotated, num, rotation_angle,
                            dir=FILE_DIR / "images/interpolation")


def interpolate_images_on_lvlset(argmin, model, img1_rotated, img2_rotated):
    """
    Interpolate between two images alongside the levelset.
    :param argmin:
    :param model:
    :param img1_rotated:
    :param img2_rotated:
    :return:
    """
    num_steps = 16

    def predict_y_from_spherical(spherical_coord, argmin):
        coord_origspace = translate_to_origin(spherical_to_cartesian(spherical_coord), argmin)
        return np.squeeze(model.decoder_y(coord_origspace).numpy())

    z1 = model.q_z(img1_rotated[tf.newaxis, ...]).mean()
    z2 = model.q_z(img2_rotated[tf.newaxis, ...]).mean()
    img1_rotated_rec = model.decoder_x(z1)
    img2_rotated_rec = model.decoder_x(z2)
    target_property = model.decoder_y(z2).numpy()
    z1_sph = cartesian_to_spherical(translate_to_minimum(z2, argmin))
    z2_sph = cartesian_to_spherical(translate_to_minimum(z1, argmin))
    delta_step = (z2_sph - z1_sph) / num_steps
    zs = [z1_sph + delta_step * i for i in range(num_steps + 1)]
    results = []
    for z in zs[1:]:
        def level_set_distance(delta_r):
            """
            Helper function that returns the deviation from the level set after adding delta_r to the radius.
            :param delta_r:
            :return:
            """
            coord_plus_r = add_radius(z.numpy(), delta_r)
            err = np.squeeze(np.abs(predict_y_from_spherical(coord_plus_r, argmin) - target_property))
            return err

        """ 
        Due to the input-convexity we can just rely on 1d optimization 
            of the radius for finding a minimum at given angles.
        """
        res = minimize_scalar(level_set_distance)
        results.append((res, add_radius(z.numpy(), res.x)))
    return img1_rotated_rec, img2_rotated_rec, results


def find_model_argmin(model: RegressionVAE,
                      max_iter: int = 10_000,
                      stepsize: float = 50.,  # 5e-1
                      clip_value_min: float = -10., clip_value_max: float = 10.
                      ):
    """
    Find argmin of the predicted Y of the model using (projected) Gradient Descent.
    This argmin will be used as the 'origin' for the parameterization of the level sets.
    :param model:
    :param max_iter:
    :param stepsize:
    :param clip_value_max:
    :param clip_value_min:
    :return:
    """

    def tmp_loss_and_gradient(x):
        return tfp.math.value_and_gradient(
            lambda x: tf.reshape(model.decoder_y(x), (1,)), x
        )

    x = tf.Variable(tf.zeros([1, model.latent_dim]))
    loss = model.decoder_y(x)
    for i in range(max_iter):
        old_loss = loss
        loss, grad = tmp_loss_and_gradient(x)
        x_proposed = x - stepsize * grad
        x = tf.clip_by_value(x_proposed,
                             clip_value_min=clip_value_min,
                             clip_value_max=clip_value_max)

        if i % 100 == 0 and i > 0:
            improvement = np.squeeze(np.abs(1 - loss / old_loss))
            # print(f"{improvement:.3e}")
            if improvement < 1e-6:
                print(f"Stopped search for minimum after {i} GD iterations.")
                break
    print(f"Found Minimum value within bounds: {np.squeeze(loss.numpy()):.2f}")
    argmin = x

    # ##### alternative using scipy:
    # from scipy import optimize
    # f = lambda x: tf.squeeze(tmp_loss_and_gradient(tf.reshape(x, (1, -1)))[0]).numpy()
    # f_ = lambda x: tf.squeeze(tmp_loss_and_gradient(tf.reshape(x, (1, -1)))[1]).numpy()
    # res = optimize.minimize(f,
    #                         jac=f_,
    #                         x0=tf.squeeze(origin.numpy()),
    #                         method='TNC',
    #                         bounds=[(-20, 20) for _ in range(ib.latent_dim)]
    #                         )
    # argmin = res.x

    return argmin


def build_networks(input_shape, label_shape) -> dict:
    """
    Returns a dictionary with all the networks required for the proposed model.

    :param input_shape:
    :return: dict(encoder_net=encoder_net,
                  encoder_mu=encoder_mu,
                  encoder_sd=encoder_sd,
                  decoder_x=decoder_x,
                  decoder_y=decoder_y)
    """
    activation = 'relu'
    # ################### ENCODER #########################
    # main branch for encoder. will be passed to mean and sd paths and passed to encoder_mu, encoder_sd
    if args.light_version:
        units = 8
    else:
        units = 32

    encoder_net = keras.Sequential([
        layers.InputLayer(input_shape),
        layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        conv2x2(units * 2, 7, strides=2),
        ConvBlock(units * 2, 3),
        conv2x2(units * 4, 3, strides=2),
        ConvBlock(units * 4, 3),
        conv2x2(units * 8, 3, strides=2),
        layers.Flatten(),
        layers.Dense(1024, activation="relu")
    ], name="Encoder")

    # Gets input by encoder_net, output must be of latent dim shape.
    encoder_mu = layers.Dense(units=args.latent_dim, name="encoder_mu")
    # Gets input by encoder_net, output must be of latent dim shape
    encoder_sd = layers.Dense(units=int(args.latent_dim * (args.latent_dim + 1) / 2),
                              name="encoder_cov")

    # ################### DECODER
    first_shape = (5, 5, units * 2 ** 5)

    decoder_x = keras.Sequential(
        [
            layers.InputLayer((args.latent_dim,)),
            layers.Dense(units=np.prod(first_shape), activation=activation),
            layers.Reshape(first_shape),
            layers.Conv2DTranspose(units * 4, 3, strides=2, padding="same"),  # 8x8 -> 16x16
            ConvBlock(units * 4, 3),
            layers.Conv2DTranspose(units * 2, 3, strides=2, padding="same"),  # 8x8 -> 16x16
            conv2x2(units * 2, 3, padding='valid'),
            layers.Conv2DTranspose(units, 3, strides=2, padding="same"),  # -> 32x32
            ConvBlock(units, 3),
            conv2x2(1, 1, activation="sigmoid"),
        ], name="Decoder_X"
    )
    decoder_x.summary()
    decoder_y = keras.Sequential(
        [
            layers.InputLayer((args.latent_dim,)),
            # activation is  softplus
            FullyInputConvexBlock(num_layers=5, units=50),
            # Only Diagsonal Weight matrix, with positive weights. Tanh Activation to get in -1 to +1 range
            layers.Dense(units=np.prod(label_shape), activation="tanh", kernel_constraint=DiagonalPositive()),
            # transform outputs to the -pi/4 to +pi/4 range
            layers.experimental.preprocessing.Rescaling(scale=np.pi / 4, offset=0)
        ], name="Decoder_y"
    )

    encoder_net.summary()
    decoder_x.summary()
    decoder_y.summary()

    return dict(encoder_net=encoder_net,
                encoder_mu=encoder_mu,
                encoder_sd=encoder_sd,
                decoder_x=decoder_x,
                decoder_y=decoder_y)


if __name__ == "__main__":
    # cmd line arguments
    parser = argparse.ArgumentParser(description="Fashion MNIST Experiment")
    parser.add_argument("--mb-size", default=32, type=int)
    parser.add_argument("--latent-dim", default=35, type=int)
    parser.add_argument("--num-epochs", default=30, type=int)
    parser.add_argument("--seed", default=1234, type=int)

    parser.add_argument('--load-model', dest='train_model', action='store_false')
    parser.add_argument('--train-model', dest='train_model', action='store_true')
    parser.set_defaults(train_model=True)

    parser.add_argument('--light-version', dest='light_version', action='store_true')
    parser.add_argument('--full-version', dest='light_version', action='store_false')
    parser.set_defaults(light_version=False)

    args = parser.parse_args()
    pprint.pformat(vars(args))

    if args.light_version:
        model_name = "model_light.h5py"
    else:
        model_name = "model.h5py"

    # logging and plot settings
    tf.get_logger().setLevel('ERROR')

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # File-Paths
    FILE_DIR = Path(__file__).resolve().parent

    os.makedirs(FILE_DIR / "images/interpolation", exist_ok=True)
    if args.train_model:
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir)
    main()
