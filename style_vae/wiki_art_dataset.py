import numpy as np
import math
import tensorflow as tf
import cv2
import glob
import random
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
from prefetch_generator import background

base_wiki_art_path = '/media/hamada/Data/Datasets/wikiart/wikiart'

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 256
CROP_SIZE = 64

def read_image(filename, dtype=tf.float32):
    f = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(f, channels=3)
    img = tf.image.convert_image_dtype( img, tf.float32)
    img = tf.cast(img, dtype)
    return img

def resize_image( img ):
    tmp_img_size = tf.random.uniform([], minval=CROP_SIZE, maxval=IMG_SIZE+1, dtype=tf.int32)
    img = tf.image.resize(img, [tmp_img_size, tmp_img_size], antialias=True)
    img = tf.image.random_crop( img, [ CROP_SIZE, CROP_SIZE, 3] )
    img = tf.image.random_flip_left_right( img )
    img = tf.image.random_flip_up_down( img )
    return img


def get_dataset(path_to_wiki_art, train=True, batch_size=32):    
    wiki_art_path = path_to_wiki_art + '/*/*.jpg'
    
    wiki_art = tf.data.Dataset.list_files( wiki_art_path, shuffle=True )
    wiki_art = wiki_art.map(lambda x: read_image(x),
                    num_parallel_calls=AUTOTUNE)
    wiki_art = wiki_art.apply( tf.data.experimental.ignore_errors() )
    wiki_art = wiki_art.map( resize_image, num_parallel_calls=AUTOTUNE)
    wiki_art = wiki_art.batch(batch_size, True)

    size = int(tf.data.experimental.cardinality(wiki_art).numpy())
    

    return wiki_art, size
    

    

if __name__ == "__main__":
   
    bs = 32
    dataset, length = get_dataset(base_wiki_art_path, train=False, batch_size=bs)
    i = 0
    last = []
    s = time.time()
    mx = 0
    pbar = tqdm(iterable=dataset, total = length)
    for wiki in pbar:
        i+=1
        m = wiki.numpy().max()
        if m > mx:
            mx = m
        pbar.set_description("Max val is: %.4f, has shape: %s" % (mx, str(wiki.shape)))

    e = time.time()
    print("Takes : ", (e-s)/length / bs, "seconds per images")
    print("MAX is: ", mx)
