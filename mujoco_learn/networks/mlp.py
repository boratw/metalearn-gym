
import numpy as np
import tensorflow as tf
import math


class MLP:
    def __init__(self, sess, name, input_dim, output_dim, hidden_sizes, output_decorate=None, learning_rate=0.001):
        self.sess = sess
        self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
        self.layer_gt = tf.placeholder(tf.float32, [None, output_dim])
        with tf.variable_scope(name):

            w1 = tf.get_variable("w1", shape=[input_dim, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (input_dim + hidden_sizes)), math.sqrt(6.0 / (input_dim + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer, dtype=tf.float32),
                trainable=True)

            fc1 = tf.matmul(self.layer_input, w1) + b1
            fc1 = tf.nn.tanh(fc1)


            w2 = tf.get_variable("w2", shape=[hidden_sizes, hidden_sizes], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + hidden_sizes)), math.sqrt(6.0 / (hidden_sizes + hidden_sizes)), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer, dtype=tf.float32),
                trainable=True)

            fc2 = tf.matmul(fc1, w2) + b2
            fc2 = tf.nn.tanh(fc2)


            w3 = tf.get_variable("w3", shape=[hidden_sizes, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes + output_dim)), math.sqrt(6.0 / (hidden_sizes + output_dim)), dtype=tf.float32),
                trainable=True)
            b3 = tf.get_variable("b3", shape=[hidden_sizes], dtype=tf.float32, 
                initializer=tf.zeros_initializer, dtype=tf.float32),
                trainable=True)

            if output_decorate is None:
                fc3 = tf.matmul(fc2, w3) + b3
            else
                fc3 = output_decorate(tf.matmul(fc2, w3) + b3)

        self.layer_output = fc3

        self.cost = tf.reduce_sum(tf.square(fc3 - self.layer_gt))
        self.operation = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        self.trainable_params = [w1, b1, w2, b2, w3, b3]

    def get_output(self, input):
        sess = tf.get_default_session()
        output = sess.run(self.layer_output, {self.layer_input:input})
        return output

    def optimize(self, input, gt):
        sess = tf.get_default_session()
        _, loss = sess.run((self.operation, self.cost), {self.layer_input:input, self.layer_gt:gt})
        return loss

    
    def add_weighted(self, source, weight):
        sess = tf.get_default_session()
        cur_param = sess.run(self.trainable_params)
        src_param = sess.run(source.trainable_params)
        
        assign_op = [ param.assign(np.add(cur_param[i] * (1.0 - weight), src_param[i] * weight)) for param in self.trainable_params ]
        sess.run(assign_op)
