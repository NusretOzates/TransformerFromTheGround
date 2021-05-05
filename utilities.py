import tensorflow as tf
import numpy as np

MAX_LENGTH = 50


@tf.function
def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
