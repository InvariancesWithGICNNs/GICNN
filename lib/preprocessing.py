"""
Includes the preprocessing functions used for rotating images.
"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from numpy import pi


def degree_to_radian(degrees):
    return 2 * pi * (degrees / 360.)


def radian_to_degree(radians):
    return (radians * 360.) / (2 * pi)


def rotate(img, img_class, angle=None, pad_size=8):
    """
    Rotate image. Uses padding to avoid cutting of parts of the objects.
    :param img:
    :param img_class:
    :param angle:
    :param pad_size:
    :return:
    """
    padding = tf.constant([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    img = img[..., tf.newaxis]
    if angle is None:
        angle = tf.random.uniform((), -np.pi / 4, np.pi / 4)

    padded_img = tf.pad(img, padding)
    new_size = pad_size - 4
    img_rotated = tfa.image.rotate(images=padded_img, angles=angle,
                                   interpolation="bilinear")[new_size:-new_size, new_size:-new_size]
    # img_rotated = padded_img[new_size:-new_size, new_size:-new_size]

    img_rotated = tf.cast(img_rotated, tf.float32) / 255.
    return img_rotated, angle[tf.newaxis], img_class
