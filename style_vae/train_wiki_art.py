import os, shutil

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
import pickle
import logging
import numpy as np
import math
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.keras.backend.clear_session()
tf.config.experimental.set_memory_growth(gpus[0],True)
tf.config.experimental.enable_tensor_float_32_execution(True)

logging.getLogger("tensorflow").setLevel(logging.INFO)

tf.random.set_seed( 7 ) #set a seed for reproducibility
np.random.seed( 7 )
from matplotlib import pyplot as plt
from perceptual_loss import perceptual_loss
from attention_layer import FastMultiHeadedAttention
from wiki_art_dataset import get_dataset
from utils import get_l2_loss
from vae import VAE, Discriminator
from tqdm import tqdm
from multiprocessing import Process
import datetime

hyperparameter_defaults = dict(
    epochs = 20,
    batch_size = 32,
    learning_rate = 1e-4,
    latent_dims=9*9,
    num_heads  = 9,
    M_N        = 1./32.,
    beta       = 4
    )

config = hyperparameter_defaults

base_wiki_art_path = '/mnt/d/Datasets/wikiart/wikiart'

train_ds, train_size = get_dataset(base_wiki_art_path, train=True , batch_size=config["batch_size"] )
val_ds, val_size   = get_dataset(base_wiki_art_path, train=False, batch_size=config["batch_size"]   )


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = './logs/' + current_time + '/all/'

for d in os.listdir('./logs/'):
    shutil.rmtree('./logs/' + d)

summary_writer = tf.summary.create_file_writer(log_dir)

def launch_tensorboard():
    import os
    os.system('tensorboard --logdir ./logs --port 8888')
    return

p = Process( target=launch_tensorboard)
p.start()

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [train_size*2, math.ceil(train_size*4)],
    [config["learning_rate"], config["learning_rate"], config["learning_rate"]*1e-2]
)
optim = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
# disc_optim = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])

def train_step(model, img):
    with tf.GradientTape() as tape, tf.GradientTape() as discTape:
        x_logit, log_var, mu = model(img)
        # l2_loss = get_l2_loss(model)
        
        generated_img = tf.nn.sigmoid( x_logit )

        mse_loss = tf.reduce_mean( tf.square( img - generated_img ) )

        raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( labels=img, logits=x_logit )

        neg_log_likelihood = tf.math.reduce_sum( raw_cross_entropy, axis=[1,2,3])

        kld_loss = - 0.5 * tf.math.reduce_sum(
            1. + tf.math.log( tf.math.square( log_var ) ) - tf.math.square( mu ) - tf.math.square( log_var ),
            axis=[1, 2, 3]
        )

        elbo = -tf.math.reduce_mean( -config['beta']*config['M_N']*kld_loss - neg_log_likelihood )

        p_loss = perceptual_loss( generated_img*255., img*255. )

        loss_total = mse_loss + elbo + p_loss
    grads = tape.gradient(loss_total, model.trainable_variables)
    grads = [ tf.clip_by_norm( g, 1.) for g in grads ]

    optim.apply_gradients(zip(grads, model.trainable_variables))
    return loss_total, elbo, mse_loss, p_loss, generated_img

def val_step(model, img):
    x_logit, log_var, mu = model(img)
    generated_img = tf.nn.sigmoid( x_logit ) 
    mse_loss = tf.reduce_mean( tf.square( img - generated_img ) )
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( labels=img, logits=x_logit )
    neg_log_likelihood = tf.math.reduce_sum( raw_cross_entropy, axis=[1,2,3])
    kld_loss = - 0.5 * tf.math.reduce_sum(
        1. + tf.math.log( tf.math.square( log_var ) ) - tf.math.square( mu ) - tf.math.square( log_var ),
        axis=[1, 2, 3]
    )
    elbo = -tf.math.reduce_mean( -config['beta']*config['M_N']*kld_loss - neg_log_likelihood )
    p_loss = perceptual_loss( generated_img*255., img*255. )
    loss_total = mse_loss + elbo + p_loss
    return loss_total, elbo, mse_loss, p_loss, generated_img

model = VAE(config["latent_dims"], config["num_heads"])

def save_model(model, name='gen'):
    weights = model.get_weights()
    if os.path.exists('./wiki_art_model/%s_backup_save.weights' % name):
        os.remove('./wiki_art_model/%s_backup_save.weights' % name)
    with open('./wiki_art_model/%s_backup_save.weights' % name, 'wb') as f:
        pickle.dump(weights, f)
    return

def load_model(model, name='gen'):
    with open('./wiki_art_model/%s_backup_save.weights' % name, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

#warmup model
model(tf.ones([1, 64, 64, 3]))

try:
    load_str = ""
    while load_str.upper() not in ["Y", "N"]:
        load_str = input("Load model? Y/N:\n")
    if load_str.upper() == "Y":
        model = load_model(model)
        # disc  = load_model(disc, 'disc')
        print("Model found, loading this model")
    elif load_str.upper() == "N":
        print("Model not loading, starting from scratch")
        save_model(model)
except Exception as e:
    print(e)
    print("Could not find model to load. Saving current model and starting from scratch")
    save_model(model)

epoch_loss_avg      = tf.keras.metrics.Mean()
epoch_mse_loss_avg  = tf.keras.metrics.Mean()
epoch_elbo_loss_avg = tf.keras.metrics.Mean()
epoch_p_loss_avg    = tf.keras.metrics.Mean()
epoch_val_loss_avg  = tf.keras.metrics.Mean()


best_loss = np.inf
t_step = tf.constant(0, dtype=tf.int64)
v_step = tf.constant(0, dtype=tf.int64)
pbar = tqdm(range(1, config["epochs"] +1))

for e in pbar:
    tf.keras.backend.clear_session()
    model = load_model(model)

    epoch_loss_avg.reset_states()
    epoch_mse_loss_avg.reset_states()
    epoch_elbo_loss_avg.reset_states()
    epoch_p_loss_avg.reset_states()
    epoch_val_loss_avg.reset_states()
    
    train_pbar = tqdm(iterable=train_ds, total=train_size, leave=True )
    for img in train_pbar:
        t_step += 1
        train_loss, elbo_loss, mse_loss, p_loss, gen_img = train_step(model, img)

        tf.debugging.check_numerics(elbo_loss , 'elbo_loss is nan')
        tf.debugging.check_numerics(mse_loss  , 'mse_loss is nan')
        tf.debugging.check_numerics(p_loss    , 'p_loss is nan')
        tf.debugging.check_numerics(train_loss, 'train_loss is nan')

        epoch_loss_avg.update_state(train_loss)
        epoch_mse_loss_avg.update_state(mse_loss)
        epoch_elbo_loss_avg.update_state(elbo_loss)
        epoch_p_loss_avg.update_state(p_loss)
        
        if t_step % 100 == 0 or t_step == 1:
            with summary_writer.as_default():
                tf.summary.scalar('learning_rate'  , lr_schedule(t_step)         , step=t_step)
                tf.summary.scalar('train_loss'     , epoch_loss_avg.result()     , step=t_step)
                tf.summary.scalar('train_mse_loss' , epoch_mse_loss_avg.result() , step=t_step)
                tf.summary.scalar('train_elbo_loss', epoch_elbo_loss_avg.result(), step=t_step)
                tf.summary.scalar('train_p_loss'   , epoch_p_loss_avg.result()   , step=t_step)
                tf.summary.image( 'train_img'      , img                         , step=t_step)
                tf.summary.image( 'train_gen_img'  , gen_img                     , step=t_step)
                summary_writer.flush()

        train_pbar.set_description( 'training loss: %.4f = %.4f + %.4f + %.4f' % (epoch_loss_avg.result(),
                        epoch_mse_loss_avg.result(), 
                        epoch_elbo_loss_avg.result(), 
                        epoch_p_loss_avg.result() ) 
        )
    
    val_pbar = tqdm(iterable=val_ds, total=val_size, leave=True)
    for img in val_pbar:
        v_step+=1
        val_loss, elbo_loss, mse_loss, p_loss, gen_img = val_step(model, img)

        epoch_val_loss_avg.update_state(val_loss)

        if v_step%50==0 or v_step == 1:
            sample = tf.random.normal(shape=[1, 8, 8, config['latent_dims']], dtype=tf.float32)
            sample = tf.nn.sigmoid( model.decode( sample ) )
            with summary_writer.as_default():
                tf.summary.scalar('val_loss'     , epoch_val_loss_avg.result()     , step=v_step)
                tf.summary.image( 'val_img'      , img                             , step=v_step)
                tf.summary.image( 'val_gen_img'  , gen_img                         , step=v_step)
                tf.summary.image('sample_gen_img', sample                          , step=v_step)
                summary_writer.flush()
        val_pbar.set_description( 'validation loss: %.4f' % epoch_val_loss_avg.result())

    cur_val_loss = epoch_val_loss_avg.result()
    if best_loss > cur_val_loss:
        best_loss = cur_val_loss
        save_model(model)

p.terminate()

print("Done!")







