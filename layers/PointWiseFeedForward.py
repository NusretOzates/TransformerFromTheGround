import tensorflow as tf


class PointWiseFeedForward(tf.keras.layers.Layer):

    def __init__(self, dimension_model, dff, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dff = dff
        self.dimension_model = dimension_model
        self.dense1 = tf.keras.layers.Dense(dff)
        self.dense2 = tf.keras.layers.Dense(dimension_model)

    def get_config(self):
        config = {
            'dff': self.dff,
            'dimension_model': self.dimension_model
        }
        base_config = super(PointWiseFeedForward, self).get_config()
        config.update(base_config)
        return config

    def call(self, inputs, **kwargs):

        result = self.dense1(inputs)
        result = self.dense2(result)

        return result
