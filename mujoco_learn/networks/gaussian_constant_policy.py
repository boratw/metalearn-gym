
import numpy as np
import tensorflow as tf
import math

from .mlp import MLP

EPS = 1e-6

class GaussianPolicy:
    def __init__(self, name, state_len, action_len, hidden_sizes=64, hidden_nonlinearity=tf.nn.relu, output_nonlinearity=tf.nn.relu, reuse=False, reg=0.001, input_tensor=None):

        with tf.variable_scope(name, reuse=reuse):
                
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, state_len])
            else:
                self.layer_input = input_tensor
            

            w1 = tf.get_variable("w1", shape=[state_len, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (state_len + hidden_sizes)), math.sqrt(6.0 / (state_len + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc1 = tf.matmul(self.layer_input, w1) + b1
            if hidden_nonlinearity is not None:
                fc1 = hidden_nonlinearity(fc1)


            w2 = tf.get_variable("w2", shape=[hidden_sizes, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + hidden_sizes)), math.sqrt(6.0 / (hidden_sizes + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc2 = tf.matmul(fc1, w2) + b2
            if hidden_nonlinearity is not None:
                fc2 = hidden_nonlinearity(fc2)


            w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + action_len)), math.sqrt(6.0 / (hidden_sizes + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc3 = tf.matmul(fc2, w3) + b3

            self.logstd_param = tf.get_variable("logstd_param", shape=[1, action_len], dtype=tf.float32, 
                initializer=tf.constant_initializer(-3.0),
                trainable=True)

            self.reg = reg

            if output_nonlinearity is not None:
                self.mu = output_nonlinearity(fc3)
                #self.logstd = tf.tile(output_nonlinearity(self.logstd_param) - 3., (tf.shape(self.mu)[0], 1))
            else:
                self.mu = fc3
                #self.logstd = tf.tile(self.logstd_param, (tf.shape(self.mu)[0], 1))
            self.logstd = tf.tile(self.logstd_param, (tf.shape(self.mu)[0], 1))
            self.std = tf.exp(self.logstd)

            #self.logsig = tf.clip_by_value(self.logsig, -20, 2)
            #self.logstd_walk = tf.clip_by_value(self.logstd, -3, 0)

            
            #ds = tf.contrib.distributions
            #self.dist = ds.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.logstd))
            #self.random_walk = ds.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.logstd - 2.0))
            #self.x = self.dist.sample()
            #self.walk = self.random_walk.sample()

            #self.log_pi = self.dist.log_prob(self.x)

            self.trainable_params = [w1, b1, w2, b2, w3, b3]

    def log_li(self, x):
        return self.dist.log_prob(x)
 

    def build_assign(self, source):
        return [ tf.assign(target, source) for target, source in zip(self.trainable_params, source.trainable_params)]
