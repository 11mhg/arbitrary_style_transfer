import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE

base_coco_path = '/mnt/d/Datasets/coco/images'

base_wiki_art_path = '/mnt/d/Datasets/wikiart/wikiart'

IMG_SIZE = 256

def read_image(filename, bfloat=False, dtype=tf.float32):
    f = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(f, channels=3)
    img = tf.image.convert_image_dtype( img, tf.float32) * 255.
    img = tf.cast(img, dtype)
    if bfloat:
        img = tf.cast(img, tf.bfloat16)
    return img

def resize_image( img ):
    img_shape = tf.cast( tf.shape(img), dtype=tf.float32)
    
    _min = tf.math.minimum( img_shape[0], img_shape[1] )
    _factor = tf.cast( 512 / _min, dtype=tf.float32)

    new_height = tf.cast( tf.math.floor( img_shape[0] * _factor ), dtype=tf.int32)
    new_width  = tf.cast( tf.math.floor( img_shape[1]  * _factor ), dtype=tf.int32)
    
    img = tf.image.resize(img, [new_height, new_width], antialias=True)
    
    if tf.random.uniform(shape=()) > 0.25:
        img = tf.image.random_crop( img, [ IMG_SIZE, IMG_SIZE, 3] )
    else:
        img = tf.image.resize( img, [IMG_SIZE, IMG_SIZE] )
    
    img = tf.image.random_flip_left_right( img )
    img = tf.image.random_flip_up_down( img )
    return img

def get_dataset(path_to_coco_base, path_to_wiki_art, data_type='train', batch_size=32, bfloat=False, dtype=tf.float32):
    coco_path = path_to_coco_base + '/' + data_type + '2017/*.jpg'
    with tf.device('/device:cpu:0'):
        coco = tf.data.Dataset.list_files( coco_path , shuffle=True)
        coco = coco.map( lambda x: read_image(x, bfloat, dtype), 
                         num_parallel_calls=AUTOTUNE)
        coco = coco.apply( tf.data.experimental.ignore_errors() )
        coco = coco.map( resize_image, num_parallel_calls=AUTOTUNE)
        coco = coco.batch( batch_size, True )
        
        wiki_art_path = path_to_wiki_art + '/*/*.jpg'
        wiki_art = tf.data.Dataset.list_files( wiki_art_path, shuffle=True )
        wiki_art = wiki_art.map(lambda x: read_image(x, bfloat, dtype),
                        num_parallel_calls=AUTOTUNE)
        wiki_art = wiki_art.apply( tf.data.experimental.ignore_errors() )
        wiki_art = wiki_art.map( resize_image, num_parallel_calls=AUTOTUNE)
        wiki_art = wiki_art.batch(batch_size, True)

        dataset = tf.data.Dataset.zip((coco, wiki_art))

    with tf.device('/device:gpu:0'):
        dataset = dataset.prefetch(AUTOTUNE)
    
    num_elem = int( np.floor( tf.data.experimental.cardinality( dataset ).numpy() ) )
    return dataset, num_elem 
    

    

if __name__ == "__main__":
   
    bs = 32
    dataset, num_elem = get_dataset(base_coco_path, base_wiki_art_path, data_type='train', batch_size=bs)
    i = 0
    last = []
    s = time.time()
    for coco, wiki in tqdm( iterable=dataset, total=num_elem) :
        i+=1 
        if i% 10 == 0:
            last = [coco.numpy(), wiki.numpy()]
            print(last[0].max(), last[1].max())
            break
    e = time.time()
    print("Takes : ", (e-s)/10 / bs, "seconds per images")
