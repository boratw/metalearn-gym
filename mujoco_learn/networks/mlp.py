
import numpy as np
import tensorflow as tf
import math


class MLP:
    def __init__(self, name, input_dim, output_dim, hidden_sizes, hidden_nonlinearity=tf.nn.relu, output_nonlinearity=None, reuse=False,
        input_tensor=None, additional_input=False, additional_input_dim=0, additional_input_tensor=None):

        with tf.variable_scope(name, reuse=reuse):

            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor
            
            w1 = tf.get_variable("w1", shape=[input_dim, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (input_dim + hidden_sizes)), math.sqrt(6.0 / (input_dim + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc1 = tf.matmul(self.layer_input, w1) + b1

            if additional_input :
                if additional_input_tensor is None:
                    self.layer_additional = tf.placeholder(tf.float32, [None, additional_input_dim])
                else:
                    self.layer_additional = additional_input_tensor
                
                w1_add = tf.get_variable("w1_add", shape=[additional_input_dim, hidden_sizes], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (additional_input_dim + hidden_sizes)), math.sqrt(6.0 / (additional_input_dim + hidden_sizes)), dtype=tf.float32),
                    trainable=True)
                b1_add = tf.get_variable("b1_add", shape=[hidden_sizes], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                

                fc1 = fc1 + tf.matmul(self.layer_additional, w1_add) + b1_add

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


            w3 = tf.get_variable("w3", shape=[hidden_sizes, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + output_dim)), math.sqrt(6.0 / (hidden_sizes + output_dim)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc3 = tf.matmul(fc2, w3) + b3
            if output_nonlinearity is not None:
                fc3 = output_nonlinearity(fc3)

            self.layer_output = fc3

            if additional_input:
                self.trainable_params = [w1, b1, w1_add, b1_add, w2, b2, w3, b3]
            else:
                self.trainable_params = [w1, b1, w2, b2, w3, b3]

    def build_add_weighted(self, source, weight):
        return [ tf.assign(target, (1 - weight) * target + weight * source) for target, source in zip(self.trainable_params, source.trainable_params)]
