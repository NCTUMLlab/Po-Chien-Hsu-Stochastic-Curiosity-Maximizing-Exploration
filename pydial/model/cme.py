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

def cosineLoss(A, B, name):
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B, 1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss

def linear(x, size, name, initializer=None, bias_init=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer) 
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init)) 
    return tf.matmul(x, w) + b

def pydialHead(x, layer2): 
    x = tf.nn.elu(linear(x, 257, 'fc', normalized_columns_initializer(0.01)))
    return x

class cme(object):
    def __init__(self, ob_space, ac_space):
        tf.reset_default_graph()
        tf.logging.set_verbosity (tf.logging.WARN) 
        self.forward_loss_wt = 0.2
        self.num_actions = ac_space
        self.num_belief_states = ob_space
        self.feat_size = 257
        self.layer2 = 257
        
        #CONFIG
        self.randomseed = 1234
        if Settings.config.has_option('GENERAL', 'seed'):
            self.randomseed = Settings.config.getint('GENERAL', 'seed')
        self.learning_rate = 0.001
        if Settings.config.has_option('scme', 'learning_rate'):
            self.learning_rate = Settings.config.getfloat('scme', 'learning_rate')

        np.random.seed(self.randomseed)
        tf.set_random_seed(self.randomseed)

        input_shape = [None, ob_space]
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = self.feat_size  
        
        phi1 = pydialHead(phi1, self.layer2)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            phi2 = pydialHead(phi2, self.layer2)

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2], 1)   # changed place of 1
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, axis=-1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat([phi1, asample], 1)
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        self.forwardloss = cosineLoss(f, phi2, name='forwardloss')

        # prediction and original
        self.predstate = f
        self.origstate = phi2


        self.predloss = self.invloss * (1 - self.forward_loss_wt) + self.forwardloss * self.forward_loss_wt

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.predloss)

        self.sess2 = tf.Session()
        self.sess2.run(tf.global_variables_initializer())
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(var_list=[v for v in all_variables if "Variab" not in v.name and "beta" not in v.name])

    def train(self, state_vec, prev_state_vec, action_1hot):
        _, predictionloss = self.sess2.run([self.optimize, self.predloss],
                                           feed_dict={self.s1: prev_state_vec,
                                           self.s2: state_vec,
                                           self.asample: action_1hot})
        return predictionloss

    def reward(self, s1, s2, asample):
        error = self.sess2.run(self.forwardloss,
                         {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        return error
        
    def predictedstate(self, s1, s2, asample):
        pred, orig = self.sess2.run([self.predstate, self.origstate],
                                    {self.s1: [s1], self.s2: [s2],
                                     self.asample: [asample]})
        return pred, orig

    def save_model(self, path_name):
        self.saver.save(self.sess2, path_name)

    def load_model(self, path_name):
        try:
            self.saver.restore(self.sess2, path_name)
        except:
            print('Not find CME model')
