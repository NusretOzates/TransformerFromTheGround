import tensorflow as tf

from layers.PointWiseFeedForward import PointWiseFeedForward
from layers.custommultiheadattention import CustomMultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dimension_model, num_heads, dff, rate=0.1, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.dimension_model = dimension_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = CustomMultiHeadAttention(dimension_model, num_heads)
        self.point_wise = PointWiseFeedForward(dimension_model, dff)

        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = {
            'dimension_model': self.dimension_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        }
        base_config = super(EncoderLayer, self).get_config()
        config.update(base_config)
        return config

    def call(self, inputs, training=None, mask=None):
        att_output = self.mha([inputs, inputs, inputs], training=training,
                              mask=mask)
        att_output = self.dropout_1(att_output, training=training)
        out_1 = self.layernorm_1(inputs + att_output)

        ffn_output = self.point_wise(out_1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out_2 = self.layernorm_2(out_1 + ffn_output)

        return out_2


class Encoder(tf.keras.Model):

    def __init__(self, num_layer: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 maximum_position_encoding: int, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=d_model,
                                                   name='EncoderEmbedding')

        # Temporary variable to create positional encodings
        self.temp_var = tf.Variable(tf.random.normal(shape=(maximum_position_encoding, d_model)), dtype=tf.float32,
                                    trainable=False)
        self.positional = self.positional_encoding(maximum_position_encoding, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layer)]
        self.dropout = tf.keras.layers.Dropout(rate)

    @tf.function
    def positional_encoding(self, position, model_dimension):
        positions = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
        dimensions = tf.range(model_dimension, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = self.get_angles(positions, dimensions, model_dimension)
        self.temp_var.assign(angle_rates)
        self.temp_var[:, 0::2].assign(tf.sin(angle_rates[:, 0::2]))
        self.temp_var[:, 1::2].assign(tf.cos(angle_rates[:, 1::2]))
        angle_rates = tf.convert_to_tensor(self.temp_var)

        pos_encoding = angle_rates[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    @tf.function
    def get_angles(self, position, dimension, dimension_model):
        angle_rates = 1 / tf.pow(tf.constant(10000, dtype=tf.float32),
                                 (2 * (dimension // 2)) / tf.constant(dimension_model, dtype=tf.float32))

        return position * angle_rates

    def get_config(self):
        config = {
            'num_layer': self.num_layer,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'input_vocab_size': self.input_vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate
        }
        base_config = super(Encoder, self).get_config()
        config.update(base_config)
        return config

    def call(self, inputs, training=None, mask=None):
        sequence_length = tf.shape(inputs)[1]
        embedding = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        f_32_d_model = tf.cast(self.d_model, tf.float32)  # cast d_model to float32
        embedding *= tf.math.sqrt(f_32_d_model)
        embedding += self.positional[:, :sequence_length, :]

        embedding = self.dropout(embedding, training=training)

        for i in range(self.num_layer):
            embedding = self.encoder_layers[i](embedding, training, mask)

        return embedding
