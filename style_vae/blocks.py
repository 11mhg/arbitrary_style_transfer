import tensorflow as tf



class Residual(tf.keras.Model):
    def __init__(self, filters, kernel, strides, padding, activation, 
                    num_layers=1, identity=True, transpose=False, 
                    merge_func=lambda x: tf.concat(x, axis=-1), 
                    name='Residual_layer'):
        super(Residual, self).__init__(name=name)
        self.conv_fun = tf.keras.layers.Conv2D if not transpose else tf.keras.layers.Conv2DTranspose
        self.cs = [ self.conv_fun(
            filters, kernel, strides=strides if i==0 else [1,1], padding=padding, activation=activation, name=name+'CONV_'+str(i)
        ) for i in range(num_layers) ]
        self.bns = [
            tf.keras.layers.BatchNormalization() for i in range(num_layers)
        ]
        if identity:
            self.r = tf.keras.layers.Lambda( lambda x: x)
        else:
            self.r = self.conv_fun(
                filters, kernel, strides=strides, padding=padding, activation=activation, name=name+'CONV_RES'
            )
        self.merge = tf.keras.layers.Lambda( merge_func )

    def call(self, x):
        residual = tf.identity(x)
        for bn, c in zip(self.bns, self.cs):
            x = c(x)
            x = bn(x)
        residual = self.r(residual)
        # residual = tf.image.resize( residual,
                        # [x.shape[1], x.shape[2]] )
        out = self.merge( [x, residual] )
        return out



class Reparameterize(tf.keras.layers.Layer):
    def __init__(self):
        super(Reparameterize, self).__init__()
        def sample(args):
            z_mean, z_log_var = args
            eps = tf.random.normal(shape=tf.shape(z_mean), mean=0., dtype=z_mean.dtype)
            ret = z_mean + tf.exp(z_log_var * .5) * eps
            # tf.print(tf.reduce_max(z_mean), "z_mean max")
            # tf.print(tf.reduce_max(z_log_var), "z_log_var max")
            # tf.print(tf.reduce_max(eps), "eps max")
            # tf.print(tf.reduce_max(ret), "ret max")
            return ret
        self. l = tf.keras.layers.Lambda(
            sample
        )
    
    def call(self, mean, logvar):
        out = self.l([mean, logvar])
        tf.debugging.check_numerics(out, "REPARAM RESULTED IN NAN OR INF")
        return out


if __name__=='__main__':
    block = Residual(32, [3,3], [1,1], 'same', tf.nn.relu, 2, identity=True, transpose=False )
    for i in range(10):
        block(tf.ones([1,32,32,3]))
    block.summary()

    r = Reparameterize()