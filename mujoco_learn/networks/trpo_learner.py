import numpy as np
import tensorflow as tf

from .mlp import MLP
from .stochastic_policy import StochasticPolicy


class TRPOLearner:
    def __init__(self, state_len, action_len, name="", policy_lr=0.01, policy_hidden_len=64, value_lr=0.01, value_hidden_len=128, value_gamma=0.96, nextstate_hidden_len=64, nextstate_lr=0.01,kl_delta=0.01):
        self.policy_reserve_cooldown = 0
        self.kl_delta = kl_delta
        with tf.variable_scope("TRPOLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_update_ratio = tf.placeholder(tf.float32, [], name="input_update_ratio")
            self.input_sigma_rate = tf.placeholder(tf.float32, [], name="input_sigma_rate")
            self.input_learning_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")

            self.value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.input_next_state)
            self.nextstate_network = MLP("nextstate", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.policy_network = StochasticPolicy("policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, input_sigma_rate=self.input_sigma_rate)

            self.new_policy_network = StochasticPolicy("new_policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu) 
            self.poilcy_initialize = [ tf.assign(target, source) 
                for target, source in zip(self.new_policy_network.trainable_params, self.policy_network.trainable_params)]

            self.value_loss = tf.reduce_mean((self.value_network.layer_output - self.input_value) ** 2)
            self.value_train = tf.train.AdamOptimizer(value_lr).minimize(loss = self.value_loss, var_list = self.value_network.trainable_params)
            self.nextstate_loss = tf.reduce_mean((self.nextstate_network.layer_output - self.input_next_state) ** 2)
            self.nextstate_train = tf.train.AdamOptimizer(nextstate_lr).minimize(loss = self.nextstate_loss, var_list = self.nextstate_network.trainable_params)


            cur_value = tf.stop_gradient(self.value_network.layer_output)
            self.next_average_state_network = MLP("nextstate", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy_network.mu, reuse=True)
            self.next_average_value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.next_average_state_network.layer_output, reuse=True)
            next_value = tf.stop_gradient(self.next_average_value_network.layer_output)
            
            self.new_policy_loss = tf.reduce_mean( -(self.input_value - next_value) * self.new_policy_network.log_prob(self.input_action) )
            self.new_policy_update = tf.train.AdamOptimizer(self.input_learning_rate).minimize(loss = self.new_policy_loss + self.new_policy_network.regularization_loss, var_list = self.new_policy_network.trainable_params) 

            self.kl_div = self.get_kl_div()

            self.policy_update = [ tf.assign(target, source)
                for target, source in zip(self.policy_network.trainable_params, self.new_policy_network.trainable_params)  ] 

    def get_kl_div(self):
        numerator = tf.square(self.policy_network.mu - self.new_policy_network.mu) + tf.square(self.policy_network.std) - tf.square(self.new_policy_network.std)
        denominator = 2 * tf.square(self.new_policy_network.std) + 1e-8
        return tf.reduce_sum( numerator / denominator + self.new_policy_network.logsig - self.policy_network.logsig)

    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.x, {self.input_state : np.array([input_state])})
        return output

    def get_action_deterministic(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.mu, {self.input_state : np.array([input_state])})
        return output

    def get_action_stochastic(self, input_state, input_sigma_rate = 1.0):
        sess = tf.get_default_session()
        output, m, s = sess.run([self.policy_network.walk,self.policy_network.mu, self.policy_network.logsig], {self.input_state : np.array([input_state]), self.input_sigma_rate : input_sigma_rate})
        return output, m, s

    def get_action_stochastic_batch(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.walk, {self.input_state : np.array(input_state), self.input_sigma_rate : input_sigma_rate})
        return output

    def optimize_value_batch(self, input_state, input_next_state, input_value):
        sess = tf.get_default_session()
        _, loss = sess.run([self.value_train, self.value_loss], {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state),  self.input_value : np.array(input_value)} )
        return loss

    def optimize_nextstate_batch(self, input_state, input_action, input_next_state):
        sess = tf.get_default_session()
        _, loss = sess.run([self.nextstate_train, self.nextstate_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action), self.input_next_state : np.array(input_next_state)} )
        return loss


    def optimize_policy_batch(self, input_learning_rate, input_state, input_action, input_value, input_next_state):
        sess = tf.get_default_session()
        sess.run(self.poilcy_initialize)
        _, loss = sess.run([self.new_policy_update, self.new_policy_loss], {self.input_learning_rate:input_learning_rate, self.input_state : np.array(input_state),  self.input_action : np.array(input_action),
            self.input_value : np.array(input_value), self.input_next_state : np.array(input_next_state)} )
        div = sess.run(self.kl_div, {self.input_state : np.array(input_state), self.input_action : np.array(input_action)})

        return loss, div
    
    def optimize_end(self):
        sess = tf.get_default_session()
        sess.run(self.policy_update)