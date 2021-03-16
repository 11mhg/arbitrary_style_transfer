import tensorflow as tf
import numpy as np


def matSqrt_tf( x ):
    D, U, V = tf.linalg.svd( x )
    result = tf.matmul( U, tf.linalg.diag( tf.math.log( D ) ) )
    result = tf.matmul( V, result, transpose_a = True)
    return result


class CORAL(tf.keras.layers.Layer):
    def __init__(self, gamma=1e-3):
        super( CORAL, self).__init__()
        self.gamma = gamma

    def call(self, inputs,alpha=1):
        source = inputs[0]
        target = inputs[1]
        
        shape = source.shape
        batch_size = shape[0]

        source = tf.reshape( source, [batch_size, shape[1]*shape[2], shape[3]] )
        target = tf.reshape( target, [batch_size, shape[1]*shape[2], shape[3]] )
        
        source_std = tf.math.reduce_std( source, axis=[0,1])
        source_mean = tf.math.reduce_mean( source, axis=[0,1])

        target_std = tf.math.reduce_std( target, axis=[0,1])
        target_mean = tf.math.reduce_mean( target, axis=[0,1])



        mc_source = ( source - source_mean )
        mc_target = ( target - target_mean )


        cov_source = (1./batch_size) * tf.matmul(mc_source, mc_source, transpose_a=True) + self.gamma * tf.eye( shape[3], shape[3] ) 
        cov_target = (1./batch_size) * tf.matmul(mc_target, mc_target, transpose_a=True) + self.gamma * tf.eye( shape[3], shape[3] ) 

        
        log_cov_source = matSqrt_tf( cov_source )
        log_cov_target = matSqrt_tf( cov_target )

        transfer = tf.linalg.inv( log_cov_source )
        transfer = tf.matmul( log_cov_target, transfer, transpose_a=True)

        normalized = tf.matmul( mc_source, transfer ) 
        normalized = normalized + source_mean

        normalized = alpha * normalized + ( 1 - alpha ) * source 

        normalized = tf.reshape( normalized, shape )

        return normalized 



if __name__ == "__main__":
    coral = CORAL()

    print( coral( [tf.ones([8, 84, 84, 3]), tf.zeros([8, 84, 84, 3])] ).shape )
