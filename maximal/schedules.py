"""
Learning Rate Schedules

- OriginalTransformerSchedule()

TODO:
add CosineWarmupSchedule()
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class OriginalTransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Source:
    https://www.tensorflow.org/text/tutorials/transformer#define_the_optimizer_with_a_custom_learning_rate_scheduler
    """
    def __init__(self, depth, warmup_steps=4000):
        super(OriginalTransformerSchedule, self).__init__()
        self.depth = tf.cast(depth, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.depth) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'depth': self.depth,
            'warmup_steps': self.warmup_steps
        }
