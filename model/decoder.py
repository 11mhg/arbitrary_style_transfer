import tensorflow as tf
from .padding import ReflectionPadding2D 
import numpy as np



class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.ls = []
        for i in range(2):
            self.ls.append(
                ReflectionPadding2D( (1,1) )
            )
            self.ls.append(
                tf.keras.layers.Conv2D( 512, [3,3], strides=1, name='conv1_block'+str(i+1) , padding='valid',
                    activation='relu')
            )

        self.ls.append(
                tf.keras.layers.UpSampling2D()
        )

        for i in range(2):
            self.ls.append(
                ReflectionPadding2D( (1,1) )
            )
            self.ls.append(
                tf.keras.layers.Conv2D( 256, [3,3], strides=1, name='conv2_block'+str(i+1) , padding='valid',
                    activation='relu')
            )

        self.ls.append(
                tf.keras.layers.UpSampling2D()
        )

        for i in range(2):
            self.ls.append(
                ReflectionPadding2D( (1,1) )
            )
            self.ls.append(
                tf.keras.layers.Conv2D( 128, [3,3], strides=1, name='conv3_block'+str(i+1) , padding='valid',
                    activation='relu')
            )

        self.ls.append(
                tf.keras.layers.UpSampling2D()
        )


        for i in range(2):
            self.ls.append(
                ReflectionPadding2D( (1,1) )
            )
            self.ls.append(
                tf.keras.layers.Conv2D( 64, [3,3], strides=1, name='conv4_block'+str(i+1) , padding='valid',
                    activation='relu')
            )
        self.ls.append(
            tf.keras.layers.Conv2D( 3, [1,1], strides=1, name='out' , padding='valid',
                activation=None)
        )







    def call(self, x):
        for l in self.ls:
            x = l(x)
        return x 




if __name__=="__main__":
    d = Decoder()

    d(tf.ones([1,32,32,512]))
