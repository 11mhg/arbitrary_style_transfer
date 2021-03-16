import sys, os, glob, shutil, re
import tensorflow as tf
import numpy as np
from model.AST import AST
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm
import argparse
tf.config.set_soft_device_placement( True )


########
#
#  PARAMS
#
#
########

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l



def reverse_preprocess( x ):
    mean = tf.constant( [103.939, 116.779, 123.68], dtype=tf.float32 )
    mean = tf.reshape( mean, [1, 1, 1, 3])
    x = x + mean
    x = x[...,::-1]
    return x

def load_weights(model):
    files = glob.glob('./weights/*.pickle')
    if len(files) > 0:
        files = sort_nicely(files)
        with open(files[-1], 'rb') as r:
            all_weights = pickle.load( r )
        model.set_weights( all_weights )
        return os.path.splitext( os.path.basename( files[-1] ) )[0]
    else:
        raise RuntimeError("Couldn't find weights!")
    return False

#build model
model = AST()
t_step = int( load_weights(model) )

model.summary()

def eval_step(content, style, alpha):
    content = model.preprocess(content)
    style   = model.preprocess(style)
    out, _, _, _, _, _ = model( [content, style], alpha=alpha ) 
    out = tf.clip_by_value( reverse_preprocess( out ), 0, 255 )
    return out

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def matSqrt(x):
    U, D, V = np.linalg.svd( x )
    result = U * (np.diag( np.power( D, 0.5 ) )) * V.T
    return result



def coral( source, target):
    H, W, _ = source.shape

    source_flatten = np.reshape( source, (-1, 3))
    target_flatten = np.reshape( target, (-1, 3))

    source_flatten = np.transpose( source_flatten, (1, 0) )
    target_flatten = np.transpose( target_flatten, (1, 0) )

    source_flatten_norm = (source_flatten - source_flatten.mean()) / source_flatten.std()
    target_flatten_norm = (target_flatten - target_flatten.mean()) / target_flatten.std()

    source_flatten_cov_eye = np.matmul( source_flatten_norm, source_flatten_norm.T ) + np.eye( 3 ).astype(np.float32)
    target_flatten_cov_eye = np.matmul( target_flatten_norm, target_flatten_norm.T ) + np.eye( 3 ).astype(np.float32)

    source_flatten_norm_transfer = np.matmul( np.matmul( matSqrt( target_flatten_cov_eye ),  np.linalg.inv( matSqrt( source_flatten_cov_eye ) ) ),  source_flatten_norm)
    source_flatten_transfer = (source_flatten_norm_transfer * target_flatten.std() ) + target_flatten.mean() 

    source_flatten_transfer = np.transpose( source_flatten_transfer, (1, 0) )
    source_flatten_transfer = np.reshape( source_flatten_transfer, (-1, 3) )

    return np.reshape( source_flatten_transfer, source.shape ).astype(np.float32)


parser = argparse.ArgumentParser(description='Process an image with an arbitrary style transfer')
parser.add_argument('--input' , type=str )
parser.add_argument('--style' , type=str )
parser.add_argument('--output', type=str )
parser.add_argument('--alpha' , type=float, default=1.)

try:
    args = parser.parse_args()

    for input_file in os.listdir( args.input ):
        for style_file in os.listdir( args.style ):

            content_img = cv2.imread( os.path.join( args.input, input_file) , cv2.IMREAD_COLOR )[:, :,::-1].astype(np.float32) 
            h, w, _     = content_img.shape
            content_img = image_resize( content_img, width=680 )
            style_img   = cv2.imread( os.path.join( args.style, style_file) , cv2.IMREAD_COLOR )[:, :,::-1].astype(np.float32)
            alpha = float(args.alpha)
            if alpha > 1 or alpha < 0:
                raise ValueError("Alpha must be between 0 and 1")
        
            style_img = cv2.resize( style_img, (content_img.shape[1], content_img.shape[0]) )
            
#            style_img = coral( style_img, content_img )

            content_img = content_img[np.newaxis, ...]
            style_img   = style_img[np.newaxis, ...]
            out         = eval_step( content_img, style_img, alpha ).numpy().astype(np.uint8)[0,:,:,::-1]
            out_img     = cv2.resize( out, (w, h) )
            input_name  = os.path.splitext( input_file)[0]
            style_name  = os.path.splitext( style_file)[0]
            cv2.imwrite( os.path.join( args.output, input_name + style_name + '.jpg'), out_img )

except IOError as msg:
    parser.error(str(msg))
