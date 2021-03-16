import tensorflow as tf
import time
from attention_layer import FastMultiHeadedAttention as FMA
from blocks import Residual


class Decoder(tf.keras.Model):
    def __init__(self, max_filters = 512, num_layers=5, num_heads=4, latent_dims=512):
        super(Decoder, self).__init__()

        self.ls = []
        min_filters = 64
        filters = list(reversed([ min(min_filters * (2**i), max_filters) for i in range(num_layers)]))
        for i in range(num_layers):
            filt = filters[i]
            strides = (i%2)+1
            self.ls.append(
                Residual(filt, [3,3], [strides,strides],
                'same', tf.nn.relu, 2, identity=False,
                transpose=True, name='Residual_layer_'+str(i) )
            )
        # self.a1 = FMA(latent_dims, num_heads)
        self.output_conv = tf.keras.layers.Conv2DTranspose(3, [3,3], [1,1], 
                                padding='same', activation='relu')
        self.final_conv  = tf.keras.layers.Conv2D(3,[1,1],padding='same',
                                activation='linear')
    
    def call(self, x):
        # a1, _ = self.a1(x, x, x)
        # x = tf.concat([a1, x], axis=-1)
        for c in self.ls:
            x = c(x)
        x = self.output_conv(x)
        x = self.final_conv(x)
        return x



if __name__=='__main__':
    e = Decoder(256, 12, latent_dims=512)

    s = time.time()
    for i in range(10):
        x = e(tf.ones([1, 4, 4, 512]))
    total = (time.time() - s)/10
    e.summary()
    print(x.shape)

    print("took: ", total )