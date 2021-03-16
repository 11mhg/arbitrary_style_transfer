import tensorflow as tf
import math



class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding=(1,1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input_spec = [ tf.keras.layers.InputSpec(ndim=4) ]

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

if __name__=="__main__":
    a = ReflectionPadding2D()
    print(a(tf.ones([1,28, 28,1])).shape)
