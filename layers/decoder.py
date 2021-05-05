import tensorflow as tf

from layers.PointWiseFeedForward import PointWiseFeedForward
from layers.custommultiheadattention import CustomMultiHeadAttention



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dimension_model, num_heads, dff, rate=0.1, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(DecoderLayer, self).__init__(trainable, name, dtype, dynamic, **kwargs)

        self.dimension_model = dimension_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.dimension_model = dimension_model
        self.num_heads = num_heads

        self.mha1 = CustomMultiHeadAttention(dimension_model, num_heads)
        self.mha2 = CustomMultiHeadAttention(dimension_model, num_heads)

        self.feed_forward = PointWiseFeedForward(dimension_model, dff)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=rate)

        self.normalizer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalizer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalizer_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):
        config = {
            'dimension_model': self.dimension_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        }
        base_config = super(DecoderLayer, self).get_config()
        config.update(base_config)
        return config

    def call(self, inputs, training=None, mask=None):
        x, enc_output, look_ahed_mask, padding_mask = inputs
        attn1 = self.mha1([x, x, x], training=training, mask=look_ahed_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.normalizer_1(attn1 + x)

        attn2 = self.mha2([enc_output, enc_output, out1], training=training, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.normalizer_2(attn2 + out1)

        feed_forward = self.feed_forward(out2)
        feed_forward = self.dropout3(feed_forward, training=training)
        out3 = self.normalizer_3(feed_forward + out2)

        return out3


class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.embedding = tf.keras.layers.Embedding(input_dim=self.target_vocab_size, output_dim=self.d_model,
                                                   name='DecoderEmbedding')
        self.temp_var = tf.Variable(tf.random.normal(shape=(maximum_position_encoding, d_model)), dtype=tf.float32,
                                    trainable=False)
        self.positional = self.positional_encoding(self.maximum_position_encoding, self.d_model)
        self.dropout = tf.keras.layers.Dropout(self.rate)

        self.decoder_layes = [DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate) for _ in
                              range(self.num_layers)]

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
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'target_vocab_size': self.target_vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'rate': self.rate
        }
        base_config = super(Decoder, self).get_config()
        config.update(base_config)
        tf.print(config)
        return config

    def call(self, inputs, training=None, mask=None):
        x, enc_output, look_ahead_mask, padding_mask = inputs
        sequence_length = tf.shape(x)[1]

        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        embedding += self.positional[:, :sequence_length, :]
        embedding = self.dropout(embedding, training=training)

        for i in range(self.num_layers):
            embedding = self.decoder_layes[i]([embedding, enc_output,
                                               look_ahead_mask, padding_mask],
                                              training=training)

        return embedding
