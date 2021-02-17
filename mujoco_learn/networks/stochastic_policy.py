
import numpy as np
import tensorflow as tf
import math

from .mlp import MLP

class StochasticPolicy:
    def __init__(self, name, state_len, action_len, hidden_sizes=64, hidden_nonlinearity=tf.nn.relu, reuse=False, reg=0.001, input_tensor=None, 
        input_sigma_rate=None, adaptive_sigma=False, init_sigma=1.0):

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

            if adaptive_sigma:
                w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len * 2], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + action_len * 2)), math.sqrt(6.0 / (hidden_sizes + action_len * 2)), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[action_len * 2], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                fc3 = tf.matmul(fc2, w3) + b3

                self.mu, self.logsig = tf.split(fc3, [action_len, action_len], 1)
                self.std = tf.exp(self.logsig)
            else:
                w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + action_len * 2)), math.sqrt(6.0 / (hidden_sizes + action_len * 2)), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[action_len], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                self.mu = tf.matmul(fc2, w3) + b3
                self.logsig = tf.get_variable("logsig", shape=[action_len], dtype=tf.float32, 
                    initializer=tf.constant_initializer(init_sigma, dtype=tf.float32),
                    trainable=True)
                self.std = tf.exp(self.logsig)

            #self.logsig = tf.clip_by_value(self.logsig, -20, 2)
            #self.logsig_walk = tf.clip_by_value(self.logsig, -5, 1)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=self.std)
            self.x = tf.clip_by_value(self.dist.sample(), -30., 30.)
            if input_sigma_rate is None:
                self.random_walk = tf.distributions.Normal(loc=self.mu, scale=self.std)
                self.walk = tf.clip_by_value(self.random_walk.sample(),-30., 30.)
            else:
                self.random_walk = tf.distributions.Normal(loc=self.mu, scale=self.std + input_sigma_rate)
                self.walk = tf.clip_by_value(self.random_walk.sample() * (30. / 30. + input_sigma_rate),-30., 30.)

            
            #self.log_pi = self.dist.log_prob(self.x)


            self.regularization_loss =  tf.reduce_mean((self.mu / 30.) ** 2 + self.logsig ** 2) * 0.0001

            self.trainable_params = [w1, b1, w2, b2, w3, b3]

    def log_prob(self, action):
        z = (action - self.mu)
        return - self.logsig - 0.5 * z ** 2


    def build_assign(self, source):
        return [ tf.assign(target, source) for target, source in zip(self.trainable_params, source.trainable_params)]
