from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import random
import math
from utils import Settings

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def linear(x,output_dim, name):
    w=tf.get_variable(name+"_w", [x.get_shape()[1], output_dim], initializer=normalized_columns_initializer(0.01))
    b=tf.get_variable(name+"_b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def encoder(x, output):
    e1 = tf.nn.tanh(linear(x, 128, 'enc_1'))
    e2 = tf.nn.tanh(linear(e1, 64, 'enc_2'))
    z_mu = linear(e2, output, 'enc_mu')
    z_log_var = linear(e2, output, 'enc_log_var')
    return z_mu, z_log_var

def encoderaction(a):
    e1 = tf.nn.tanh(linear(a, 16, 'enc_a_1'))
    e2 = tf.nn.tanh(linear(e1, 12, 'enc_a_2'))
    za_mu = linear(e2, 8, 'za_mu')
    za_log_var = linear(e2, 8, 'za_log_var')
    return za_mu, za_log_var

def reparameterization(z_mu, z_log_var):
    eps = tf.random_normal(shape=tf.shape(z_log_var), mean=0, stddev=1, dtype=tf.float32)
    z = z_mu + tf.exp(0.5 * z_log_var) * eps
    return z

def decoder(z, output):
    with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
        d1 = tf.nn.tanh(linear(z, 64, 'dec_1'))
        d2 = tf.nn.tanh(linear(d1, 128, 'dec_2'))
        x = tf.nn.sigmoid(linear(d2, output, 'dec_3'))
    return x

def decoderaction(z, output):
    with tf.variable_scope('decoderaction',reuse=tf.AUTO_REUSE):
        d1 = tf.nn.tanh(linear(z, 16, 'deca_1'))
        x = tf.nn.sigmoid(linear(d1, output, 'deca_2'))
    return x

def kl_div_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    '''Batched KL divergence D(q||p) computation.'''
    kl= 0.5*tf.reduce_sum(((p_mu - q_mu)/tf.exp(0.5*p_logvar))**2,1) + \
        tf.reduce_sum(0.5*p_logvar,1) - tf.reduce_sum(0.5*q_logvar,1) + \
        0.5*tf.reduce_sum((tf.exp(0.5*q_logvar)/tf.exp(0.5*p_logvar))**2,1) - 0.5
    return tf.reduce_mean(kl)

def cosineLoss(A, B):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B, 1)), 1)
    loss = 1-tf.reduce_mean(dotprod)
    return loss

def reduce_logmeanexp_nodiag(x, axis=None):
    batch_size = x.shape[0].value
    logsumexp = tf.reduce_logsumexp(x - tf.diag(np.inf * tf.ones(batch_size)), axis=axis)
    if axis:
        num_elem = batch_size - 1.
    else:
        num_elem  = batch_size * (batch_size - 1.)
    return logsumexp - tf.log(num_elem)

def mine(scores):
    return tf.reduce_mean(tf.linalg.diag_part(scores)) - reduce_logmeanexp_nodiag(scores)


class scme(object):
    def __init__(self, ob_space, ac_space):
        tf.reset_default_graph()
        tf.logging.set_verbosity (tf.logging.WARN)
        self.num_actions = ac_space 
        self.num_belief_states = ob_space
        self.z_space = 16
    
        #CONFIG
        self.randomseed = 1234
        if Settings.config.has_option('GENERAL', 'seed'):
            self.randomseed = Settings.config.getint('GENERAL', 'seed')
        self.learning_rate = 0.001
        if Settings.config.has_option('scme', 'learning_rate'):
            self.learning_rate = Settings.config.getfloat('scme', 'learning_rate')
        self.gamma_s = 1.0
        if Settings.config.has_option('scme', 'gamma_s'):
            self.gamma_s = Settings.config.getfloat('scme', 'gamma_s')
        self.gamma_a = 1.0
        if Settings.config.has_option('scme', 'gamma_a'):
            self.gamma_a = Settings.config.getfloat('scme', 'gamma_a')
        self.gamma_c = 1.0
        if Settings.config.has_option('scme', 'gamma_c'):
            self.gamma_c = Settings.config.getfloat('scme', 'gamma_c')
        self.gamma_m = 1.0
        if Settings.config.has_option('scme', 'gamma_m'):
            self.gamma_m = Settings.config.getfloat('scme', 'gamma_m')

        np.random.seed(self.randomseed)
        tf.set_random_seed(self.randomseed)

        input_shape = [None, ob_space]
        self.s1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = tf.placeholder(tf.float32, input_shape)
        self.asample = tf.placeholder(tf.float32, [None, ac_space])

        #encode:
        z1_mu, z1_log_var = encoder(self.s1, self.z_space)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            z2_mu, z2_log_var = encoder(self.s2, self.z_space)
        za_mu, za_log_var = encoderaction(self.asample)

        #reparameterization
        self.z1 = reparameterization(z1_mu, z1_log_var)
        self.z2 = reparameterization(z2_mu, z2_log_var)
        self.za = reparameterization(za_mu, za_log_var)

        #decode:
        s1_hat = decoder(self.z1, self.num_belief_states)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            s2_hat = decoder(self.z2, self.num_belief_states)
        a_hat = decoderaction(self.za, self.num_actions)
        
        #vae_loss:
        self.recon1_loss = tf.reduce_mean(-tf.reduce_sum(self.s1 * tf.log(s1_hat) + (1 - self.s1) * tf.log(1 - s1_hat) , 1))
        self.z1_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z1_mu) + tf.exp(z1_log_var) - z1_log_var - 1 , 1))
        self.recon2_loss = tf.reduce_mean(-tf.reduce_sum(self.s2 * tf.log(s2_hat) + (1 - self.s2) * tf.log(1 - s2_hat) , 1))
        self.z2_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z2_mu) + tf.exp(z2_log_var) - z2_log_var - 1 , 1))
        self.recona_loss = tf.reduce_mean(-tf.reduce_sum(self.asample * tf.log(a_hat) + (1 - self.asample) * tf.log(1 - a_hat) , 1))
        self.za_kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(za_mu) + tf.exp(za_log_var) - za_log_var - 1 , 1))
        self.vae_loss_s = self.recon2_loss + self.z2_kl_loss
        self.vae_loss_a = self.recona_loss + self.za_kl_loss 
        
        #curiosity network: 
        c = tf.concat([self.z1, self.za], 1)
        c1 = tf.nn.relu(linear(c, 24 , 'cur_1'))
        c2 = tf.nn.relu(linear(c1, 16 , 'cur_2'))
        c_mu = linear(c2, self.z_space, 'cur_mu')
        c_log_var = linear(c2, self.z_space, 'cur_log_sigma')
        self.cur_loss = kl_div_gaussian(z2_mu, z2_log_var, c_mu, c_log_var)
        self.c_z = reparameterization(c_mu, c_log_var)
        self.cur_rew = cosineLoss(self.c_z, self.z2)

        #decode predicted latent state
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.predstate = decoder(self.c_z, self.num_belief_states)
        self.origstate = self.s2

        #information network
        batch_size = 128
        # Tile all possible combinations of x and y
        x_tiled = tf.tile(self.c_z[None, :],  (batch_size, 1, 1))
        y_tiled = tf.tile(self.za[:, None],  (1, batch_size, 1))
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = tf.reshape(tf.concat([x_tiled, y_tiled], 2), [batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        mi1 = tf.nn.relu(linear(xy_pairs, 128, 'mi_za_11'))
        mi2 = tf.nn.relu(linear(mi1, 128, 'mi_za_21'))
        mi3 = linear(mi2, 1, 'mi_za_31')
        mi = tf.transpose(tf.reshape(mi3, [batch_size, batch_size]))
        self.mi_loss = mine(mi)
        
        self.total_loss = self.gamma_s * self.vae_loss_s + self.gamma_a * self.vae_loss_a + self.gamma_c * self.cur_loss - self.gamma_m * self.mi_loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.total_loss)
        self.sess2 = tf.Session()
        self.sess2.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, state_vec, prev_state_vec, action_1hot):
        #make each batch be equal
        prev_state_vec = prev_state_vec[0:128]
        state_vec = state_vec[0:128]
        action_1hot = action_1hot[0:128]
        _, totalloss = self.sess2.run([self.optimize, self.total_loss],
                                        feed_dict={self.s1: prev_state_vec,
                                        self.s2: state_vec,
                                        self.asample: action_1hot})
        return totalloss
    
    def reward1(self, s1, s2, asample):
        ri1 = self.sess2.run(self.cur_rew, {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        return ri1
    
    def reward2(self, s1, s2, asample):
        #make each step be equal to a batchsize
        s1 = s1[None,:]
        s2 = s2[None,:]
        s11 = s1
        s22 = s2
        asample = asample[None,:]
        act = asample
        for i in range(127):
            s11 = np.vstack((s1,s11))
            s22 = np.vstack((s2,s22))
            act = np.vstack((asample,act))
        ri2 = self.sess2.run(self.mi_loss, {self.s1: s11, self.s2: s22, self.asample: act})
        return ri2

    def latent_code(self, s1, s2, asample):
        pred_zs, orig_zs, pred_s, orig_s = self.sess2.run([self.c_z, self.z2, self.predstate , self.origstate],
                         {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        return pred_zs, orig_zs, pred_s, orig_s

    def save_model(self, path_name):
        self.saver.save(self.sess2, path_name)
    
    def load_model(self, path_name):
        try:
            self.saver.restore(self.sess2, path_name)
        except:
            print()




