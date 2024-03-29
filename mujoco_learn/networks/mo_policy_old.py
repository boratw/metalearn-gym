
import numpy as np
import tensorflow as tf
import math

from .mlp import MLP

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

            w3 = tf.get_variable("w3", shape=[hidden_sizes, action_len], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + action_len)), math.sqrt(6.0 / (hidden_sizes + action_len)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[action_len], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
                
            fc3 = tf.matmul(fc2, w3) + b3

            self.layer_output_action = tf.nn.tanh(fc3)
            #self.layer_output_action = tf.clip_by_value(fc3, -action_maximum, action_maximum)
            self.dist = tf.distributions.Normal(loc=self.layer_output_action, scale=action_maximum * 0.1)


            self.random_foggy = tf.distributions.Normal(loc=self.layer_output_action, scale=action_maximum * 0.05)
            self.layer_foggy_action = tf.clip_by_value(self.random_foggy.sample(), -action_maximum, action_maximum)

            self.random_foggy2 = tf.distributions.Normal(loc=self.layer_output_action, scale=action_maximum * 0.1)
            self.layer_foggy2_action = tf.clip_by_value(self.random_foggy2.sample(), -action_maximum, action_maximum)

            self.random_diverse = tf.distributions.Normal(loc=self.layer_output_action * 0.3, scale=action_maximum * 0.5)
            self.layer_diverse_action = tf.clip_by_value(self.random_diverse.sample(), -action_maximum, action_maximum)

            self.random_collect = tf.distributions.Normal(loc=self.layer_output_action * 0.8, scale=action_maximum * 0.1)
            self.layer_collect_action = tf.clip_by_value(self.random_collect.sample(), -action_maximum, action_maximum)

            self.trainable_params = [w1, b1, w2, b2, w3, b3]

            self.regularization_loss =  0.01 * tf.reduce_sum( tf.reduce_mean(fc3 ** 2) )
    
    def get_dist(self, action_std):
        return  tf.distributions.Normal(loc=self.layer_output_action, scale=action_std).sample()

    def log_prob(self, action):
        return self.dist.log_prob(action)
