
import numpy as np
import tensorflow as tf
import math

from .mlp import MLP

EPS = 1e-6

class ModelBasedPolicy:
    def __init__(self, name, state_len, action_len, hidden_sizes=64, hidden_nonlinearity=tf.nn.tanh, reuse=False, 
        input_tensor=None, action_maximum = 1.0):

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

            w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + action_len)), math.sqrt(6.0 / (hidden_sizes + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len * 2], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
                
            fc3 = tf.matmul(fc2, w3) + b3
            self.mu, self.logsig = tf.split(fc3, [action_len, action_len], 1)
            self.std = tf.exp(self.logsig)
            self.clipped_mu = tf.tanh(self.mu)
            self.logsig_collect = tf.clip_by_value(self.logsig, -4, 0) * 0.5 + 1.
            self.logsig_diverse = tf.clip_by_value(self.logsig, -4, 0) * 0.5 + 2.

            ds = tf.contrib.distributions
            self.dist = ds.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.logsig))
            #self.random_action = ds.MultivariateNormalDiag(loc=self.clipped_mu, scale_diag=tf.exp(self.logsig))
            self.random_collect = ds.MultivariateNormalDiag(loc=self.clipped_mu, scale_diag=tf.exp(self.logsig_collect))
            self.random_diverse = ds.MultivariateNormalDiag(loc=self.clipped_mu * 0.5, scale_diag=tf.exp(self.logsig_diverse))

            self.x = self.dist.sample()
            self.layer_output_action = tf.tanh(self.x)
            self.layer_diverse_action = tf.clip_by_value(self.random_diverse.sample(), -1.0, 1.0)
            self.layer_collect_action = tf.clip_by_value(self.random_collect.sample(), -1.0, 1.0)


            self.trainable_params = [w1, b1, w2, b2, w3, b3]

            self.log_pi = self.dist.log_prob(self.x) - self.squash_correction(self.layer_output_action)
            self.regularization_loss = 0.001 * tf.reduce_sum( tf.reduce_mean(self.mu ** 2) + tf.reduce_mean(self.logsig ** 2))
    
    def log_prob(self, action):
        return self.dist.log_prob(action)

    def squash_correction(self, actions):
        return tf.reduce_sum(tf.log(1 - actions ** 2 + EPS), axis=1)
