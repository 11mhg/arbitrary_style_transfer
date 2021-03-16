import tensorflow as tf
import numpy as np

from .encoder import Encoder
from .decoder import Decoder
from .adain import AdaIN



class AST(tf.keras.Model):
    def __init__(self):
        super( AST, self).__init__()

        self.encoder = Encoder()
        self.encoder.trainable = False
        self.adain   = AdaIN()
        self.decoder = Decoder()
        self.decoder.trainable = True

        #preheat model
        _ = self( [ tf.ones([16,256,256,3]), tf.ones([16, 256, 256, 3]) ] )
    
    def preprocess(self,x):
        return self.encoder.preprocess( x )

    def call(self, inputs, alpha=1):
        content = inputs[0]
        style   = inputs[1]
        c_out, c_feats = self.encoder( content )
        s_out, s_feats = self.encoder( style )
        
        t = self.adain( [c_out, s_out], alpha=alpha )

        m_out = self.decoder( t )

        return m_out, t, c_out, c_feats, s_out, s_feats


if __name__=='__main__':
    ast = AST()

    o, t, c1, c2, s1, s2 = ast( tf.ones([1, 256, 256, 3]), tf.zeros([1, 256, 256, 3]) )

    print(o.shape, t.shape, c1.shape, s1.shape)

