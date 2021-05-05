import tensorflow as tf


class TransformerLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss_ = loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
