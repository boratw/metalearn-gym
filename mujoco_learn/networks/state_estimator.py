
import numpy as np
import tensorflow as tf
import math


class StateEstimator:
    def __init__(self, name, state_dim, action_dim, hidden_sizes, hidden_nonlinearity=tf.nn.tanh, reuse=False,
        input_state_tensor=None, input_action_tenser=False):

        with tf.variable_scope(name, reuse=reuse):

            if input_state_tensor is None:
                self.layer_input_state = tf.placeholder(tf.float32, [None, state_dim])
            else:
                self.layer_input_state = input_state_tensor

            if input_action_tenser is None:
                self.layer_input_action = tf.placeholder(tf.float32, [None, action_dim])
            else:
                self.layer_input_action = input_action_tenser
            
            w1_1 = tf.get_variable("w1_1", shape=[state_dim, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (state_dim + hidden_sizes)), math.sqrt(6.0 / (state_dim + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1_1 = tf.get_variable("b1_1", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc1_1 = tf.matmul(self.layer_input_state, w1_1) + b1_1

                
            w1_2 = tf.get_variable("w1_2", shape=[action_dim, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (action_dim + hidden_sizes)), math.sqrt(6.0 / (action_dim + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1_2 = tf.get_variable("b1_2", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
                
            fc1_2 = tf.matmul(self.layer_input_action, w1_2) + b1_2

            fc1 = tf.concat([fc1_1, fc1_2], axis=1)

            if hidden_nonlinearity is not None:
                fc1 = hidden_nonlinearity(fc1)


            w2 = tf.get_variable("w2", shape=[hidden_sizes * 2 , hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes * 3)), math.sqrt(6.0 / (hidden_sizes * 3)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc2 = tf.matmul(fc1, w2) + b2
            if hidden_nonlinearity is not None:
                fc2 = hidden_nonlinearity(fc2)


            w3 = tf.get_variable("w3", shape=[hidden_sizes , hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes * 2)), math.sqrt(6.0 / (hidden_sizes * 2)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc3 = tf.matmul(fc2, w3) + b3
            if hidden_nonlinearity is not None:
                fc3 = hidden_nonlinearity(fc3)

            w4 = tf.get_variable("w4", shape=[hidden_sizes, state_dim + 1], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + state_dim + 1)), math.sqrt(6.0 / (hidden_sizes + state_dim + 1)), dtype=tf.float32),
                trainable=True)
            b4 = tf.get_variable("b4", shape=[state_dim + 1], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc4 = tf.matmul(fc3, w4) + b4

            self.layer_output_state, self.layer_output_reward = tf.split(fc4, [state_dim, 1], 1)
            self.random_state = tf.distributions.Normal(loc=self.layer_output_state, scale=self.layer_output_state * 0.01)
            self.layer_foggy_state = self.random_state.sample()

            self.trainable_params = [w1_1, b1_1, w1_2, b1_2, w2, b2, w3, b3, w4, b4]

