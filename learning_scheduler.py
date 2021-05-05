import tensorflow as tf


class CustomScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_states': self.warmup_steps
        }
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.cast(tf.math.rsqrt(tf.cast(self.d_model, tf.float32)), tf.float32) * tf.cast(
            tf.math.minimum(arg1, arg2), tf.float32)
