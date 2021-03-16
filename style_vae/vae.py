import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from blocks import Reparameterize
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dims=512, num_heads=4):
        super(VAE, self).__init__()

        self.latent_dims = latent_dims
        self.encoder = Encoder(256, 6, num_heads, self.latent_dims)
        self.decoder = Decoder(256, 6, num_heads, self.latent_dims)
        self.reparam = Reparameterize()
    
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = tf.split(x, 2, axis=-1)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparam(mean, logvar)
        x_logit = self.decode(z)
        return x_logit, logvar, mean


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c1 = tf.keras.layers.Conv2D(64, [7,7], strides=2, padding='same',
                                activation=tf.nn.leaky_relu)
        self.d1 = tf.keras.layers.Dropout(0.3)
        self.c2 = tf.keras.layers.Conv2D(128, [3,3], strides=2, padding='same',
                                activation=tf.nn.leaky_relu)
        self.d2 = tf.keras.layers.Dropout(0.3)
        self.c3 = tf.keras.layers.Conv2D(256, [3,3], strides=2, padding='same',
                                activation=tf.nn.leaky_relu)
        self.d3 = tf.keras.layers.Dropout(0.3)
        
        self.F = tf.keras.layers.Flatten()
        self.D = tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu)
        self.O = tf.keras.layers.Dense(1)

    def call(self, x, training=None):
        x = self.c1(x)
        x = self.d1(x, training)
        x = self.c2(x)
        x = self.d2(x, training)
        x = self.c3(x)
        x = self.d3(x, training)

        x = self.F(x)
        x = self.D(x)
        logits = self.O(x)

        probs = tf.nn.sigmoid(logits)
        return probs, logits
    
if __name__=="__main__":
    vae = VAE(64)

    vae(tf.random.normal([32, 64, 64, 3]))

    vae.summary()


    d = Discriminator()

    p, l = d(tf.ones([1, 64, 64, 3]))

    print(p.shape, l.shape)
