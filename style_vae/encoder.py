import tensorflow as tf
import time
from attention_layer import FastMultiHeadedAttention as FMA
from blocks import Residual

class Encoder(tf.keras.Model):
    def __init__(self, max_filters = 512, num_layers=5, num_heads = 8, latent_dims=512):
        super(Encoder, self).__init__()

        self.ls = []
        min_filters = 64
        filters = [ min(min_filters * (2**i), max_filters) for i in range(num_layers)]
        for i in range(num_layers):
            filt = filters[i]
            strides = (i%2)+1
            self.ls.append(
                Residual(filt, [3,3], [strides,strides],
                'same', tf.nn.relu, 2, identity=False, name='Residual_layer_'+str(i) )
            )
        # self.a1 = FMA(max_filters, num_heads)
        self.output_conv = tf.keras.layers.Conv2D(latent_dims*2, [1,1], [1,1], padding='same', activation='linear')
    
    def call(self, x):
        for c in self.ls:
            x = c(x)
        # a1, _ = self.a1(x, x, x)
        # x = tf.concat([x, a1], axis=-1)
        x = self.output_conv(x)
        return x


if __name__=='__main__':
    e = Encoder(256, 12, latent_dims=256)

    s = time.time()
    for i in range(10):
        x = e(tf.ones([1, 256, 256, 3]))
    total = (time.time() - s)/10
    e.summary()
    print(x.shape)