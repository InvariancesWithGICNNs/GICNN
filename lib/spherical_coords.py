"""
Includes functions for dealing with spherical coordinates.
"""
import numpy as np
import tensorflow as tf


def cartesian_to_spherical(arr):
    """

    :param arr:
    :return:
    """
    radius = tf.sqrt(tf.reduce_sum(arr ** 2, axis=1))[..., tf.newaxis]
    sqrt_sums = tf.sqrt(tf.cumsum(arr ** 2, axis=1, reverse=True))[:, :-1]
    angles = tf.acos(arr[:, :-1] / sqrt_sums)
    last_angle = tf.cast(arr[:, -1:] >= 0, 'float') * angles[:, -1] + \
                 tf.cast(arr[:, -1:] < 0, 'float') * (2 * np.pi - angles[:, -1])

    return tf.concat([radius, angles[:, :-1], last_angle], -1)


def spherical_to_cartesian(arr):
    """

    :param arr:
    :return:
    """
    r = arr[:, 0]
    angles = arr[:, 1:]
    sin_prods = tf.math.cumprod(tf.sin(angles), axis=1)
    x1 = r * tf.cos(angles[..., :1])
    xs = r * sin_prods[..., :-1] * tf.cos(angles[..., 1:])
    xn = r * sin_prods[..., -1:]
    return tf.concat([x1, xs, xn], -1)


def translate_to_minimum(coord, argmin):
    """
    :param coord:
    :param argmin:
    :return:
    """
    return coord - argmin


def translate_to_origin(coord, argmin):
    """

    :param coord:
    :param argmin:
    :return:
    """
    return coord + argmin


def add_radius(spherical_coord, r):
    """
    Add radius to spherical coordinate
    :param spherical_coord:
    :param r:
    :return:
    """
    tmp = spherical_coord.copy()
    tmp[:, 0] += np.squeeze(r)
    return tmp