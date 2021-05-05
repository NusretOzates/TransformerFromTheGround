import tensorflow as tf


class CustomMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, dimension_model, num_heads, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_heads = num_heads
        self.dimension_model = dimension_model

        # Because we will divide the query,key and value by num_heads!
        assert dimension_model % num_heads == 0

        self.depth = dimension_model // self.num_heads

        self.weight_query = tf.keras.layers.Dense(dimension_model, activation='elu')
        self.weight_key = tf.keras.layers.Dense(dimension_model, activation='elu')
        self.weight_value = tf.keras.layers.Dense(dimension_model, activation='elu')

        self.dense = tf.keras.layers.Dense(dimension_model, activation='elu')

    # It is a multi attention!

    @tf.function
    def attention(self, inputs):
        query, key, value, mask = inputs[0], inputs[1], inputs[2], inputs[3]
        # Matmul query and key, scale it, add softmax and matmul with value.
        matmul_query_key = tf.matmul(query, key, transpose_b=True)
        # Generally 64 for multihead-attention and 512 for self-attention, according to the paper
        dimension_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_query_key / tf.math.sqrt(dimension_key)

        if mask is not None:
            # I have no clue what it does... debug time
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)  # (...,seq_len_q, depth_v)

        return output

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'dimension_model': self.dimension_model,
        }
        base_config = super(CustomMultiHeadAttention, self).get_config()
        config.update(base_config)
        return config

    @tf.function
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size,num_heads,seq_len,depth) -> (64,8,512,64)
        """
        # (batch_size,variable_length_column, num_heads,depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None, mask=None):
        value, key, query = inputs[0], inputs[1], inputs[2]
        batch_size = tf.shape(query)[0]

        query = self.weight_query(query)  # (batch_size, seq_len, d_model)
        key = self.weight_key(key)  # (batch_size, seq_len, d_model)
        value = self.weight_value(value)  # (batch_size, seq_len, d_model)

        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size,num_heads, seq_len_query, depth)
        # attention_weights.shape == (batch_size,num_heads, seq_len_query, seq_len_k)

        scaled_attention = self.attention([query, key, value, mask])
        # [0,2,1,3] means, new matrices' dimension will be (0.th dim of old matrix, second dimension of old matrix...)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        """
          >>> x = tf.constant(
          [
              [
                  [ 1,  2,  3],
                  [ 4,  5,  6]
              ],
              [
                  [ 7,  8,  9],
                  [10, 11, 12]
              ]
          ]) = (2,2,3)


          >>> tf.transpose(x, perm=[0, 2, 1])
  <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
  array(
  [
      [
          [ 1,  4],
          [ 2,  5],
          [ 3,  6]
      ],
      [
          [ 7, 10],
          [ 8, 11],
          [ 9, 12]
      ]
  ], dtype=int32)>



        """

        # By using reshape with "-1" we are co
        # cating the heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dimension_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, sequence_length_query, dimension_model)
        return output
