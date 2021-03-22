import numpy as np
import tensorflow as tf
import random

from .state_estimator import StateEstimator
from .mo_policy import ModelBasedPolicy


class MORLLearner:
    def __init__(self, state_len, action_len, name="", policy_lr=0.001, policy_hidden_len=64, state_lr=0.01, state_hidden_len=128, gamma=0.95, action_maximum=1.0):
        self.action_maximum = action_maximum
        with tf.variable_scope("MORLLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_advantage = tf.placeholder(tf.float32, [None, 1], name="input_advantage")
            self.input_learning_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")
            self.input_action_std = tf.placeholder(tf.float32, [action_len], name="input_action_std")

            self.state_network = StateEstimator("state", state_len, action_len, state_hidden_len, 
                input_state_tensor=self.input_state, input_action_tenser=self.input_action, hidden_nonlinearity=tf.nn.leaky_relu)
            self.policy_network = ModelBasedPolicy("policy", state_len, action_len, policy_hidden_len, 
                input_tensor=self.input_state, action_maximum=action_maximum, hidden_nonlinearity=tf.nn.leaky_relu)
            self.nextstate_network = StateEstimator("state", state_len, action_len, state_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, 
                input_state_tensor=self.input_state, input_action_tenser=self.policy_network.layer_foggy_action, reuse=True)
            self.nextstate_sto_network = StateEstimator("state", state_len, action_len, state_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, 
                input_state_tensor=self.input_state, input_action_tenser=self.policy_network.layer_collect_action, reuse=True)


            self.new_policy_network = ModelBasedPolicy("new_policy", state_len, action_len, policy_hidden_len, 
                input_tensor=self.input_state, action_maximum=action_maximum, hidden_nonlinearity=tf.nn.leaky_relu)
            self.poilcy_initialize = [ tf.assign(target, source) 
                for target, source in zip(self.new_policy_network.trainable_params, self.policy_network.trainable_params)]

            self.state_loss = tf.reduce_mean((self.state_network.layer_output_state - self.input_next_state) ** 2)
            self.reward_loss = tf.reduce_mean((self.state_network.layer_output_reward - self.input_reward) ** 2)
            self.state_train = tf.train.AdamOptimizer(state_lr).minimize(loss = self.state_loss + self.reward_loss, var_list = self.state_network.trainable_params)

            self.policy_loss = tf.reduce_mean( ((self.new_policy_network.layer_output_action - self.input_action) ** 2) * (1. / (self.input_action_std + 1e-6)) * self.input_advantage)
            self.policy_train = tf.train.GradientDescentOptimizer(self.input_learning_rate).minimize(loss = self.policy_loss, var_list = self.new_policy_network.trainable_params)

            self.kl_div = tf.reduce_mean(tf.square(self.policy_network.layer_output_action - self.new_policy_network.layer_output_action) )
            self.policy_update = [ tf.assign(target, source)
                for target, source in zip(self.policy_network.trainable_params, self.new_policy_network.trainable_params)  ] 


    def optimize_nextstate_batch(self, input_state, input_action, input_reward, input_next_state):
        sess = tf.get_default_session()
        _, ls, lr = sess.run([self.state_train, self.state_loss, self.reward_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_reward : np.array(input_reward), self.input_next_state : np.array(input_next_state)} )
        return ls, lr

    def optimize_policy_batch(self, input_learning_rate, input_action_std, input_state, input_action, input_advantage):
        sess = tf.get_default_session()
        sess.run(self.poilcy_initialize)
        _, loss = sess.run([self.policy_train, self.policy_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_advantage : np.array(input_advantage), self.input_learning_rate : input_learning_rate, self.input_action_std : input_action_std} )
        div = sess.run(self.kl_div, {self.input_state : np.array(input_state), self.input_action : np.array(input_action), self.input_action_std : input_action_std})
        return loss, div


    def get_next(self, input_state, input_action):
        sess = tf.get_default_session()
        state, reward = sess.run([self.state_network.layer_output_state, self.state_network.layer_output_reward],
                {self.input_state : np.array([input_state]), self.input_action : np.array([input_action])} )
        return state[0], reward[0][0]

    def get_next_diff(self, input_state, input_action, input_next_state):
        sess = tf.get_default_session()
        ls = sess.run(self.state_loss, {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_next_state : np.array(input_next_state)} )
        return ls

    def get_next_with_policy(self, input_state):
        sess = tf.get_default_session()
        state, reward = sess.run([self.nextstate_network.layer_output_state, self.nextstate_network.layer_output_reward],
                {self.input_state : np.array([input_state])} )
        return state[0], reward[0][0]


    def get_nexts(self, input_state, input_action):
        sess = tf.get_default_session()
        state, reward = sess.run([self.state_network.layer_output_state, self.state_network.layer_output_reward],
                {self.input_state : np.array(input_state), self.input_action : np.array(input_action)} )
        return state, np.reshape(reward, -1)

    def get_nexts_with_policy(self, input_state):
        sess = tf.get_default_session()
        state, reward = sess.run([self.nextstate_network.layer_output_state, self.nextstate_network.layer_output_reward],
                {self.input_state : np.array(input_state)} )
        return state, np.reshape(reward, -1)

    def get_nexts_sto_with_policy(self, input_state):
        sess = tf.get_default_session()
        state, reward = sess.run([self.nextstate_sto_network.layer_output_state, self.nextstate_sto_network.layer_output_reward],
                {self.input_state : np.array(input_state)} )
        return state, np.reshape(reward, -1)



    def get_actions_diverse(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_diverse_action,
                {self.input_state : np.array(input_state)} )
        
        return action

    def get_action_collecting(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_collect_action,
                {self.input_state : np.array([input_state])} )
        
        return action[0]
    

    def get_action_diverse(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_diverse_action,
                {self.input_state : np.array([input_state])} )
        
        return action[0]

    def get_action_stochastic(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_foggy_action,
                {self.input_state : np.array([input_state])} )
        return action[0]
        

    def get_action_optimal(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_output_action,
                {self.input_state : np.array([input_state])} )
        
        return action[0]

    def optimize_end(self):
        sess = tf.get_default_session()
        sess.run(self.policy_update)