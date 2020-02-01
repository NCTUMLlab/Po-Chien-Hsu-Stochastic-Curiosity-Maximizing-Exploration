from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import random
import math
from utils import Settings

class vime(object):
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

        np.random.seed(self.randomseed)
        tf.set_random_seed(self.randomseed)

        self.batch_num = 64
        self.act_func = tf.nn.sigmoid
        self.stddev = 0.1
        self.stddev_prior = tf.exp(-3.0)
        with tf.variable_scope('vime', reuse = False): 
            self.batch_idx = 2
            input_shape = [None, ob_space]
            self.s1 = tf.placeholder(tf.float32, input_shape)
            self.s2 = tf.placeholder(tf.float32, input_shape)
            self.asample = tf.placeholder(tf.float32, [None, ac_space])

            # Prior of weights and biases
            self.W1_mean = tf.get_variable('_W1_mean', initializer = tf.truncated_normal([ob_space+ac_space,128],stddev = self.stddev))  
            self.W1_logstd = tf.get_variable('_W1_logstd', initializer = tf.truncated_normal([ob_space+ac_space,128],stddev = self.stddev)) 
                                      
            self.b1_mean = tf.get_variable('_b1_mean', initializer = tf.truncated_normal([128],stddev = self.stddev))  
            self.b1_logstd = tf.get_variable('_b1_logstd', initializer = tf.truncated_normal([128],stddev = self.stddev))

            self.W1_noise = tf.random_normal([ob_space+ac_space,128], mean = 0., stddev = self.stddev_prior) 
            self.b1_noise = tf.random_normal([128], mean = 0., stddev = self.stddev_prior)               
          
            self.W2_mean = tf.get_variable('_W2_mean', initializer = tf.truncated_normal([128,ob_space],stddev = self.stddev))  
            self.W2_logstd = tf.get_variable('_W2_logstd', initializer = tf.truncated_normal([128,ob_space],stddev = self.stddev))              

            self.b2_mean = tf.get_variable('_b2_mean', initializer = tf.truncated_normal([ob_space],stddev = self.stddev))  
            self.b2_logstd = tf.get_variable('_b2_logstd', initializer = tf.truncated_normal([ob_space],stddev = self.stddev))
          
            self.W2_noise = tf.random_normal([128,ob_space], mean = 0., stddev = self.stddev_prior) 
            self.b2_noise = tf.random_normal([ob_space], mean = 0., stddev = self.stddev_prior) 
          
            self.bbn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'vime')
          
               # Weights and biases
            self.W1 = self.W1_mean + tf.multiply(tf.log(1. + tf.exp(self.W1_logstd)), self.W1_noise)   
            self.b1 = self.b1_mean + tf.multiply(tf.log(1. + tf.exp(self.b1_logstd)), self.b1_noise)   
          
            self.W2 = self.W2_mean + tf.multiply(tf.log(1. + tf.exp(self.W2_logstd)), self.W2_noise) 
            self.b2 = self.b2_mean + tf.multiply(tf.log(1. + tf.exp(self.b2_logstd)), self.b2_noise)             
          
               # Connection
            self.h1 = self.act_func(tf.add(tf.matmul(tf.concat([self.s1, self.asample], 1), self.W1), self.b1))
            self.pred = tf.add(tf.matmul(self.h1, self.W2), self.b2)
          
               # Loss function
            self.sample_log_pw = tf.reduce_sum(self.log_gaussian(self.W1,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.b1,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.W2,0.,self.stddev_prior)) \
                                  + tf.reduce_sum(self.log_gaussian(self.b2,0.,self.stddev_prior))
          
            self.sample_log_qw = tf.reduce_sum(self.log_gaussian_logstd(self.W1,self.W1_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.b1,self.b1_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.W2,self.W2_mean,self.stddev_prior*2)) \
                                  + tf.reduce_sum(self.log_gaussian_logstd(self.b2,self.b2_mean,self.stddev_prior*2))
          
            self.sample_log_likelihood = tf.reduce_sum(self.log_gaussian(self.s2,self.pred,self.stddev_prior))
            self.pi = (2**(self.batch_num-self.batch_idx-1))/(2**self.batch_num-1)
          
            self.total_loss = tf.reduce_mean((self.sample_log_qw - self.sample_log_pw) - self.sample_log_likelihood)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, var_list = self.bbn_vars)

            self.sess2 = tf.Session()
            self.sess2.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()    
          
          
    def train(self, state_vec, prev_state_vec, action_1hot):
        _, totalloss = self.sess2.run([self.optimizer, self.total_loss],
                                           feed_dict={self.s1: prev_state_vec,
                                           self.s2: state_vec,
                                           self.asample: action_1hot})
        return totalloss

    def get_hyper_parameters(self):
         
        hyper_parameters = self.sess2.run([[self.W1_mean, self.W2_mean], 
                                       [self.W1_logstd,self.W2_logstd], 
                                       [self.b1_mean, self.b2_mean], 
                                       [self.b1_logstd,self.b2_logstd]])
          
        return hyper_parameters 
      
    def get_info_gain(self,hyper_parameters,pre_hyper_parameters):	 
		  
        W_means = hyper_parameters[0]
        W_logstds = hyper_parameters[1]
        b_means = hyper_parameters[2]  
        b_logstds = hyper_parameters[3]   
                   
        pre_W_means = pre_hyper_parameters[0]
        pre_W_logstds = pre_hyper_parameters[1]
        pre_b_means = pre_hyper_parameters[2]  
        pre_b_logstds = pre_hyper_parameters[3]   

        length = len(pre_W_means)
        total_div = 0.
        for l in range(length):   
            W_div = self.KL_div(W_means[l],W_logstds[l],pre_W_means[l],pre_W_logstds[l])  
            b_div = self.KL_div(b_means[l],b_logstds[l],pre_b_means[l],pre_b_logstds[l])   
            total_div = total_div + W_div + b_div
        
        return total_div/length
          
    def KL_div(self,mean_1,log_std_1,mean_2,log_std_2):
        term_1 = np.mean((np.exp(log_std_1)/np.exp(log_std_2))**2) 
        term_2 = np.mean(2*log_std_2-2*log_std_1)
        term_3 = np.mean(((mean_1-mean_2)/log_std_2)**2)
        return 0.5*(term_1+term_2+term_3-1)

    def log_gaussian(self,x,mean,std):
          
        return -0.5*np.log(2*np.pi)-tf.log(std)-((x-mean)**2)/(2*(std**2)) 
      
    def log_gaussian_logstd(self,x,mean,logstd):   

        return -0.5*np.log(2*np.pi)-logstd/2.-((x-mean)**2)/(2.*tf.exp(logstd)) 
    
    def save_model(self, path_name):
        self.saver.save(self.sess2, path_name)
    
    def load_model(self, path_name):
        try:
            self.saver.restore(self.sess2, path_name)
        except:
            print('Not find VIME model')



