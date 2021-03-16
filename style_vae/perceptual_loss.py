import tensorflow as tf
import numpy as np




RESNET_MODEL = tf.keras.applications.ResNet50V2( include_top=False, weights='imagenet' )
RESNET_MODEL.trainable = False
out_layer_names = ['conv5_block3_out', "conv5_block2_out", "conv5_block1_out"]
out_layers = [ RESNET_MODEL.get_layer( i ).output for i in out_layer_names]
perceptual_model = tf.keras.Model([RESNET_MODEL.input], outputs=out_layers )
perceptual_model.trainable = False


def perceptual_loss( y_pred, y_true):
    y_pred = tf.keras.applications.resnet_v2.preprocess_input( y_pred )
    y_true = tf.keras.applications.resnet_v2.preprocess_input( y_true )

    all_pred_feats = perceptual_model( y_pred )
    all_true_feats = perceptual_model( y_true )

    all_pred_feats = tf.concat( all_pred_feats, axis=0 )
    all_true_feats = tf.concat( all_true_feats, axis=0 )

    p_loss = tf.square( all_true_feats - all_pred_feats)
    p_loss = tf.reduce_sum( p_loss, axis= [1,2,3])
    p_loss = tf.reduce_mean( p_loss )

    return p_loss





if __name__=='__main__':
    y_true = tf.ones([10, 84, 84, 3], dtype=tf.float32) * 255.
    y_pred = tf.ones([10, 84, 84, 3], dtype=tf.float32)

    print( perceptual_loss( y_pred, y_true ))
