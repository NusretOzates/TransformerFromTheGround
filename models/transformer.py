import tensorflow as tf

from layers.decoder import Decoder
from layers.encoder import Encoder


class MaskingLayer(tf.keras.layers.Layer):
    """
    Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
    """

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    @tf.function
    def create_padding_mask(self, sequence):
        # if the number equal to 0 , returns True else False
        """
        tf.Tensor(
        [[False False  True  True False]
         [False False False  True  True]
         [True  True  True False False]], shape=(3, 5), dtype=bool)
        """""
        sequence = tf.math.equal(sequence, 0)
        # Cast boolean to float32
        sequence = tf.cast(sequence, tf.float32)

        # add extra dimensions to add the padding to the attention logits.
        return sequence[:, tf.newaxis, tf.newaxis, :]  # (batchsize,1,1,sequence_length)

    @tf.function
    def create_look_ahead_mask(self, size: int):
        ones = tf.ones((size, size), dtype=tf.float32)
        """
         tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
        """
        mask = 1 - tf.linalg.band_part(ones, -1, 0)
        return mask

    def call(self, inputs, **kwargs):
        encoder_input, target = inputs[0], inputs[1]

        encoder_padding_mask = self.create_padding_mask(encoder_input)

        decoder_padding_mask = self.create_padding_mask(encoder_input)

        look_ahead_mask = self.create_look_ahead_mask(tf.shape(target)[1])
        decoder_target_padding_mask = self.create_padding_mask(target)
        combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

        return encoder_padding_mask, combined_mask, decoder_padding_mask


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')


class Transformer(tf.keras.Model):
    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'pe_input': self.pe_input,
            'pe_target': self.pe_target,
            'rate': self.rate,
            'EOS_TOKEN': self.EOS_TOKEN
        }
        base_config = super(Transformer, self).get_config()
        config.update(base_config)
        return config

    def train_step(self, data):
        inp, target = data
        tar_real = target[:, 1:]
        target = target[:, :-1]
        mask = self.calculate_masked_loss(tar_real)

        with tf.GradientTape() as tape:
            preds = self([inp, target], True)
            loss = self.compiled_loss(tar_real, preds, sample_weight=mask) / tf.reduce_sum(mask)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        train_loss.update_state(loss)
        acc = self.accuracy_function(tar_real, preds)
        train_accuracy.update_state(acc)

        return {'loss': train_loss.result(), 'accuracy': train_accuracy.result()}

    def test_step(self, data):
        inp, target = data
        tar_real = target[:, 1:]
        target = target[:, :-1]
        mask = self.calculate_masked_loss(tar_real)

        preds = self([inp, target], True)
        loss = self.compiled_loss(tar_real, preds) * mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        val_loss.update_state(loss)
        val_accuracy.update_state(self.accuracy_function(tar_real, preds))
        return {'loss': val_loss.result(), 'accuracy': val_accuracy.result()}

    @property
    def metrics(self):
        return [train_loss, train_accuracy, val_loss, val_accuracy]

    @tf.function
    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    @tf.function
    def calculate_masked_loss(self, tar_real):
        mask = tf.math.logical_not(tf.math.equal(tar_real, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        return mask

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, EOS_TOKEN, rate=0.1):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pe_input = pe_input
        self.pe_target = pe_target
        self.rate = rate
        self.EOS_TOKEN = EOS_TOKEN

        self.masker = MaskingLayer()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        encoder_input, target = inputs[0], inputs[1]

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.masker([encoder_input, target])
        enc_output = self.encoder(encoder_input, training=training, mask=enc_padding_mask)
        dec_output = self.decoder([target, enc_output, look_ahead_mask,
                                   dec_padding_mask], training=training, mask=None)
        final = self.final_layer(dec_output)

        return final
