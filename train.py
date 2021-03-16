import sys, os, glob, shutil, re
os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
import tensorflow as tf
import numpy as np
from datasets import get_dataset
from model.AST import AST
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm

########
#
#  PARAMS
#
#
########

batch_size = 8 
style_weights = 1e-2
learning_rate = 1e-4
learning_rate_decay = 5e-5
beta_1 = 0.9
max_iter = 160000


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


def save_weights(model, step):
    all_weights = model.get_weights()
    files = glob.glob('./weights/*.pickle')
    files = sort_nicely(files)
    if len(files) > 3:
        for f in files[:2]: 
            os.remove( f )
    with open('./weights/%d.pickle'%(step), 'wb') as w:
        pickle.dump( all_weights, w )
    return

def load_weights(model):
    files = glob.glob('./weights/*.pickle')
    if len(files) > 0:
        files = sort_nicely(files)
        with open(files[-1], 'rb') as r:
            all_weights = pickle.load( r )
        model.set_weights( all_weights )
        return os.path.splitext( os.path.basename( files[-1] ) )[0]
    return False



base_coco_path = '/mnt/d/Datasets/coco/images'
base_wiki_art_path = '/mnt/d/Datasets/wikiart/wikiart'
dataset, num_elem = get_dataset(base_coco_path, base_wiki_art_path, data_type='train', batch_size = batch_size)

#build model
model = AST()

#for content, style in dataset.take(1):
#    plt.imshow( content[0,...]/255. )
#    plt.show()
#    encoded = model.preprocess( content )
#    decoded = reverse_preprocess( encoded )
#    plt.imshow( decoded[0, ...] / 255. )
#    plt.show()
if len(glob.glob('./weights/*.pickle')) == 0:
    save_weights(model, 0)

t_step = int( load_weights(model) )

model.summary()

rate = tf.keras.optimizers.schedules.InverseTimeDecay( learning_rate, 1, learning_rate_decay )
opt = tf.keras.optimizers.Adam(learning_rate = rate, beta_1 = beta_1)

opt.iterations.assign( tf.constant( t_step, dtype=tf.int64 ) )

total_loss_mean = tf.keras.metrics.Mean()
loss_c_mean     = tf.keras.metrics.Mean()
loss_s_mean     = tf.keras.metrics.Mean()


if os.path.exists('./logs/'):
    shutil.rmtree('./logs')

os.mkdir('./logs')
writer = tf.summary.create_file_writer('./logs/')


def train_step(content, style):
    content = model.preprocess(content)
    style   = model.preprocess(style)
    bs      = tf.cast( tf.shape(content)[0], dtype=tf.float32 )
    with tf.GradientTape() as tape:
        out, t, c_out, c_feats, s_out, s_feats = model( [content, style] )

        pred_t, pred_feats = model.encoder( out )

        l_c = tf.reduce_mean(
            tf.square(
                t - pred_t
            )
        )

        l_s = 0. 
        for ind in range(len(pred_feats)):
            p = pred_feats[ind]
            s = s_feats[ind]
            s_mean, s_var = tf.nn.moments( s, [1,2] )
            p_mean, p_var = tf.nn.moments( p, [1,2] )
            m_loss = tf.reduce_sum(
                tf.square(
                    s_mean - p_mean
                )
            ) / bs
            s_loss = tf.reduce_sum(
                tf.square(
                    tf.sqrt( s_var ) - tf.sqrt( p_var )
                )
            ) / bs
            m_std_loss = m_loss + s_loss
            m_std_loss = style_weights * m_std_loss
            l_s        = l_s + m_std_loss

        total_loss = l_c + l_s 

    grads = tape.gradient( total_loss, model.trainable_variables )
    opt.apply_gradients( zip(grads, model.trainable_variables ) )
    
    out = reverse_preprocess( out ) 
    return total_loss, l_c, l_s, out

def val_step( content, style ):
    content = model.preprocess(content)
    style   = model.preprocess(style)
    
    out, t, c_out, c_feats, s_out, s_feats = model( [content, style] )

    pred_t, pred_feats = model.encoder( out )

    l_c = tf.reduce_mean(
        tf.square(
            t - pred_t
        )
    )

    l_s = 0. 
    for ind in range(len(pred_feats)):
        p = pred_feats[ind]
        s = s_feats[ind]
        s_mean, s_var = tf.nn.moments( s, [1,2] )
        p_mean, p_var = tf.nn.moments( p, [1,2] )
        m_loss = tf.reduce_sum(
            tf.square(
                s_mean - p_mean
            )
        ) / bs
        s_loss = tf.reduce_sum(
            tf.square(
                tf.sqrt( s_var ) - tf.sqrt( p_var )
            )
        ) / bs
        m_std_loss = m_loss + s_loss
        m_std_loss = style_weights * m_std_loss
        l_s        = l_s + m_std_loss

    total_loss = l_c + l_s

    out = reverse_preprocess( out )
    return total_loss, l_c, l_s, out


pbar_ep = tqdm( range(1, max_iter+1 ) )
for _ in range(t_step):
    pbar_ep.update(1)

best_loss = np.inf

while t_step < max_iter:
    load_weights(model)

    total_loss_mean.reset_states()
    loss_c_mean.reset_states()
    loss_s_mean.reset_states()
    
    for coco, wiki in dataset:
        pbar_ep.update(1)
        t_step += 1
        total_loss, l_c, l_s, out_img = train_step( coco, wiki )
        total_loss_mean.update_state( total_loss )
        loss_c_mean.update_state( l_c )
        loss_s_mean.update_state( l_s )

        if t_step % 20 == 0:
            pbar_ep.set_description("%d - %.4f = %.4f + %.4f" %( t_step, total_loss_mean.result().numpy(),
                loss_c_mean.result().numpy(), loss_s_mean.result().numpy() ))
            with writer.as_default():
                tf.summary.scalar('total_loss'      ,total_loss_mean.result()  ,step=t_step)
                tf.summary.scalar('content_loss'    ,loss_c_mean.result()      ,step=t_step)
                tf.summary.scalar('style_loss'      ,loss_s_mean.result()      ,step=t_step)
                tf.summary.scalar('learning_rate'   ,rate(t_step)              ,step=t_step)
                tf.summary.image('style_image'      ,wiki/255.                 ,step=t_step)
                tf.summary.image('content_image'    ,coco/255.                 ,step=t_step)
                tf.summary.image('generated_image'  ,out_img/255.              ,step=t_step) 
        if t_step % 200 == 0:
            loss_val = total_loss_mean.result().numpy()
            if best_loss >= loss_val:
                save_weights( model, t_step )

