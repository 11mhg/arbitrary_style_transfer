import tensorflow as tf
import numpy as np

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
 
        vgg19 = tf.keras.applications.VGG19( include_top = False, weights='imagenet')
        output = [vgg19.get_layer('block'+str(i)+'_conv1').output for i in range(1, 5)]
        inputs = [vgg19.input]
        model = tf.keras.Model( inputs = inputs, outputs=output)
        self.model = model

    def preprocess(self, x):
        return tf.keras.applications.vgg19.preprocess_input( x )

    def call(self, x):
        feats = self.model(x)
        return feats[-1], feats

if __name__=='__main__':
    fe = Encoder()

    print([ i.shape for i in fe(tf.ones([1,256,256,3]) )[-1] ])
