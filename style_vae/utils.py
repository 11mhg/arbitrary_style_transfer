import tensorflow as tf


def get_l2_loss(model, scale=1e-1):
    model_vars = model.trainable_variables
    l = [ tf.reduce_sum(tf.square(i))/2. for i in model_vars ]
    return tf.add_n(l) * scale