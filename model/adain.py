import tensorflow as tf
import numpy as np


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, data_format='channels_last'):
        super( AdaIN, self).__init__() 
        self.axes = [1,2] if data_format == 'channels_last' else [2, 3]
        self.epsilon = epsilon

    def call(self, inputs,alpha=1):
        content = inputs[0]
        style   = inputs[1]
        c_mean, c_var = tf.nn.moments( content, axes = self.axes, keepdims = True)
        s_mean, s_var = tf.nn.moments( style  , axes = self.axes, keepdims = True)
        c_std , s_std = tf.sqrt( c_var + self.epsilon), tf.sqrt( s_var + self.epsilon )
        normalized    = s_std * (content - c_mean ) / c_std + s_mean

        normalized = alpha * normalized + ( 1 - alpha ) * content

        return normalized 



if __name__ == "__main__":
    ain = AdaIN()

    print( ain(tf.ones([1, 84, 84, 3]), tf.zeros([1, 84, 84, 3]) ).shape )
