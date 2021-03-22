import numpy as np
import tensorflow as tf
import random

from .mlp import MLP
from .state_estimator import StateEstimator
from .mo_policy import ModelBasedPolicy


class MORLLearner:
    def __init__(self, state_len, action_len, name="", 
        policy_lr=0.001, policy_hidden_len=128, 
        value_lr = 0.001, value_hidden_len = 256, 
        qvalue_lr = 0.001, qvalue_hidden_len = 256, 
        state_lr=0.01, state_hidden_len=128, 
        gamma=0.95, action_maximum=1.0, update_kl_div=1e-2):

        self.action_maximum = action_maximum
        self.learning_rate = 0.01
        self.update_kl_div = update_kl_div
        with tf.variable_scope("MORLLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_qvalue = tf.placeholder(tf.float32, [None, 1], name="input_qvalue")
            self.input_survival_gamma = tf.placeholder(tf.float32, [None, 1], name="input_survival_gamma")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")

            self.state_network = StateEstimator("state", state_len, action_len, state_hidden_len, 
                input_state_tensor=self.input_state, input_action_tenser=self.input_action, hidden_nonlinearity=tf.nn.leaky_relu)
            self.qvalue1_network = MLP("qvalue1", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.qvalue2_network = MLP("qvalue2", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.value_network = MLP("value", state_len, 1, value_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state)
            self.value_next_network = MLP("value_next", state_len, 1, value_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_next_state)
            self.policy_network = ModelBasedPolicy("policy", state_len, action_len, policy_hidden_len, 
                input_tensor=self.input_state, action_maximum=action_maximum, hidden_nonlinearity=tf.nn.leaky_relu)

            self.nextstate_network = StateEstimator("state", state_len, action_len, state_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, 
                input_state_tensor=self.input_state, input_action_tenser=self.policy_network.layer_output_action, reuse=True)
            self.nextstate_optimal_network = StateEstimator("state", state_len, action_len, state_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, 
                input_state_tensor=self.input_state, input_action_tenser=self.policy_network.clipped_mu, reuse=True)
            self.qvalue1_policy_network = MLP("qvalue1", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, reuse=True,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy_network.layer_output_action )
            self.qvalue2_policy_network = MLP("qvalue2", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, reuse=True,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy_network.layer_output_action )

            self.state_loss = tf.reduce_mean((self.state_network.layer_output_state - self.input_next_state) ** 2)
            self.reward_loss = tf.reduce_mean((self.state_network.layer_output_reward - self.input_reward) ** 2)
            self.state_train = tf.train.AdamOptimizer(state_lr).minimize(loss = self.state_loss + self.reward_loss, var_list = self.state_network.trainable_params)

            soft_value = self.input_qvalue + self.input_survival_gamma * tf.stop_gradient(self.value_next_network.layer_output)
            self.qvalue1_loss = tf.reduce_mean((soft_value - self.qvalue1_network.layer_output) ** 2)
            self.qvalue1_train = tf.train.AdamOptimizer(qvalue_lr).minimize(loss = self.qvalue1_loss, var_list = self.qvalue1_network.trainable_params)
            self.qvalue2_loss = tf.reduce_mean((soft_value - self.qvalue2_network.layer_output) ** 2)
            self.qvalue2_train = tf.train.AdamOptimizer(qvalue_lr).minimize(loss = self.qvalue2_loss, var_list = self.qvalue2_network.trainable_params)
            self.mean_qvalue = tf.reduce_mean(self.qvalue1_network.layer_output)

            self.policy_loss = tf.reduce_mean( self.policy_network.log_pi -self.qvalue1_policy_network.layer_output)
            self.policy_train = tf.train.AdamOptimizer(policy_lr).minimize(loss = self.policy_loss + self.policy_network.regularization_loss,
                var_list = self.policy_network.trainable_params)
            self.mean_policy = tf.reduce_mean(self.policy_network.log_pi)

            min_log_target = tf.minimum(self.qvalue1_policy_network.layer_output, self.qvalue2_policy_network.layer_output)
            self.value_loss = tf.reduce_mean((self.value_network.layer_output - tf.stop_gradient(min_log_target - self.policy_network.log_pi)) ** 2) 
            self.value_train = tf.train.AdamOptimizer(value_lr).minimize(loss = self.value_loss,
                var_list = self.value_network.trainable_params)
            self.mean_value = tf.reduce_mean(self.value_network.layer_output)

            self.value_update = [ tf.assign(target, target * 0.9 + source * 0.1)
                for target, source in zip(self.value_next_network.trainable_params, self.value_network.trainable_params)  ] 


    def optimize_nextstate_batch(self, input_state, input_action, input_reward, input_next_state):
        sess = tf.get_default_session()
        _, ls, lr = sess.run([self.state_train, self.state_loss, self.reward_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_reward : np.array(input_reward), self.input_next_state : np.array(input_next_state)} )
        return ls, lr

    def optimize_batch(self, input_state, input_action, input_qvalue, input_survival_gamma, input_next_state):
        sess = tf.get_default_session()
        _, _, _, _, q, v, p = sess.run([self.qvalue1_train, self.qvalue2_train, self.value_train, self.policy_train,
            self.mean_qvalue, self.mean_value, self.mean_policy ],
            {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_qvalue : input_qvalue, self.input_survival_gamma : input_survival_gamma, self.input_next_state : input_next_state} )
        sess.run(self.value_update)
        return q, v, p

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

    def get_nexts_with_policy_optimal(self, input_state):
        sess = tf.get_default_session()
        state, reward = sess.run([self.nextstate_optimal_network.layer_output_state, self.nextstate_optimal_network.layer_output_reward],
                {self.input_state : np.array(input_state)} )
        return state, np.reshape(reward, -1)



    def get_qvalues(self, input_state, input_action):
        sess = tf.get_default_session()
        qvalue = sess.run(self.qvalue_network.layer_output,
                {self.input_state : np.array(input_state), self.input_action : np.array(input_action) } )
        
        return np.reshape(qvalue, -1)

    def get_qvalues_with_policy(self, input_state):
        sess = tf.get_default_session()
        qvalue = sess.run(self.qvalue_policy_network.layer_output,
                {self.input_state : np.array(input_state)} )
        
        return np.reshape(qvalue, -1)

    def get_actions_diverse(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_diverse_action,
                {self.input_state : np.array(input_state)} )
        
        return action

    def get_actions_optimal(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_output_action,
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


    def get_action_optimal(self, input_state):
        sess = tf.get_default_session()
        action = sess.run(self.policy_network.layer_output_action,
                {self.input_state : np.array([input_state])} )
        
        return action[0]
