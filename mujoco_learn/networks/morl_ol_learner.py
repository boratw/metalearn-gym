import numpy as np
import tensorflow as tf
import random

from .mlp import MLP
from .state_estimator import StateEstimator
from .mo_policy import ModelBasedPolicy


class MORLLearner:
    def __init__(self, state_len, action_len, name="", 
        policy_lr=0.001, policy_hidden_len=64, 
        qvalue_lr = 0.001, qvalue_hidden_len = 128, 
        state_lr=0.01, state_hidden_len=128, 
        gamma=0.95, action_maximum=1.0, update_kl_div=1e-2):

        self.state_lr = state_lr
        self.qvalue_lr = qvalue_lr
        self.scope = "MORLLearner" + name

        self.action_maximum = action_maximum
        self.learning_rate = 0.01
        self.update_kl_div = update_kl_div
        with tf.variable_scope(self.scope): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_qvalue = tf.placeholder(tf.float32, [None, 1], name="input_qvalue")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_advantage = tf.placeholder(tf.float32, [None, 1], name="input_advantage")
            self.input_learning_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")
            self.input_action_std = tf.placeholder(tf.float32, [action_len], name="input_action_std")

            self.state_network = StateEstimator("state", state_len, action_len, state_hidden_len, 
                input_state_tensor=self.input_state, input_action_tenser=self.input_action, hidden_nonlinearity=tf.nn.leaky_relu)
            self.qvalue_network = MLP("qvalue", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.policy_network = ModelBasedPolicy("policy", state_len, action_len, policy_hidden_len, 
                input_tensor=self.input_state, action_maximum=action_maximum, hidden_nonlinearity=tf.nn.leaky_relu)

            self.nextstate_network = StateEstimator("state", state_len, action_len, state_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, 
                input_state_tensor=self.input_state, input_action_tenser=self.policy_network.layer_output_action, reuse=True)
            self.new_policy_network = ModelBasedPolicy("new_policy", state_len, action_len, policy_hidden_len, 
                input_tensor=self.input_state, action_maximum=action_maximum, hidden_nonlinearity=tf.nn.leaky_relu)
            self.poilcy_initialize = [ tf.assign(target, source) 
                for target, source in zip(self.new_policy_network.trainable_params, self.policy_network.trainable_params)]
            self.qvalue_policy_network = MLP("qvalue", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, reuse=True,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.new_policy_network.layer_output_action )

            self.state_loss = tf.reduce_mean((self.state_network.layer_output_state - self.input_next_state) ** 2)
            self.reward_loss = tf.reduce_mean((self.state_network.layer_output_reward - self.input_reward) ** 2)
            self.state_train = tf.train.AdamOptimizer(self.input_learning_rate).minimize(loss = self.state_loss + self.reward_loss, var_list = self.state_network.trainable_params)

            self.qvalue_loss = tf.reduce_mean((self.input_qvalue - self.qvalue_network.layer_output) ** 2)
            self.qvalue_train = tf.train.AdamOptimizer(self.input_learning_rate).minimize(loss = self.qvalue_loss, var_list = self.qvalue_network.trainable_params)

            self.policy_loss = tf.reduce_mean( self.new_policy_network.log_pi -self.qvalue_policy_network.layer_output + self.new_policy_network.regularization_loss)
            self.policy_train = tf.train.AdamOptimizer(self.input_learning_rate).minimize(loss = self.policy_loss, var_list = self.new_policy_network.trainable_params)

            self.kl_div = self.get_kl_div()
            self.policy_update = [ tf.assign(target, source)
                for target, source in zip(self.policy_network.trainable_params, self.new_policy_network.trainable_params)  ] 

    def get_kl_div(self):
        numerator = tf.square(self.policy_network.mu - self.new_policy_network.mu) + tf.square(self.policy_network.std) - tf.square(self.new_policy_network.std)
        denominator = 2 * tf.square(self.new_policy_network.std) + 1e-8
        return tf.reduce_sum( numerator / denominator + self.new_policy_network.logsig - self.policy_network.logsig)

    def optimize_nextstate_batch(self, learning_rate, input_state, input_action, input_reward, input_next_state):
        sess = tf.get_default_session()
        _, ls, lr = sess.run([self.state_train, self.state_loss, self.reward_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_reward : np.array(input_reward), self.input_next_state : np.array(input_next_state), self.input_learning_rate : learning_rate * self.state_lr  } )
        return ls, lr

    def optimize_qvalue_batch(self, learning_rate, input_state, input_action, input_qvalue):
        sess = tf.get_default_session()
        _, loss = sess.run([self.qvalue_train, self.qvalue_loss], {self.input_state : np.array(input_state), self.input_action : np.array(input_action),
                self.input_qvalue : input_qvalue, self.input_learning_rate : learning_rate * self.qvalue_lr} )
        return loss

    def optimize_policy_batch(self, learning_rate, input_state):
        sess = tf.get_default_session()
        overfitted = False
        max_div = self.update_kl_div * learning_rate
        while True:
            sess.run(self.poilcy_initialize)
            _, loss = sess.run([self.policy_train, self.policy_loss], {self.input_state : np.array(input_state), self.input_learning_rate : self.learning_rate} )
            div = sess.run(self.kl_div, {self.input_state : np.array(input_state)})
            print("Policy_lr: " + str(self.learning_rate) + " Loss:" + str(loss) + " Divergence:" + str(div) )
            if div > max_div or np.isnan(div):
                overfitted = True
                self.learning_rate /= 2.
            else:
                if overfitted:
                    break
                else:
                    self.learning_rate *= 2.
        sess.run(self.policy_update)
        return loss

    def optimize_end(self):
        sess = tf.get_default_session()
        sess.run(self.policy_update)


    def get_next(self, input_state, input_action):
        sess = tf.get_default_session()
        state, reward = sess.run([self.state_network.layer_output_state, self.state_network.layer_output_reward],
                {self.input_state : np.array([input_state]), self.input_action : np.array([input_action])} )
        return state[0], reward[0][0]

    def get_next_diff(self, input_state, input_action, input_next_state):
        sess = tf.get_default_session()
        ls = sess.run(self.state_loss, {self.input_state : np.array([input_state]), self.input_action : np.array([input_action]),
                self.input_next_state : np.array([input_next_state])} )
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

    def optimize_end(self):
        sess = tf.get_default_session()
        sess.run(self.policy_update)

    def init_from_maml(self, source):
        sess = tf.get_default_session()
        state_update = [ tf.assign(target, source)
                for target, source in zip(self.state_network.trainable_params, source.state_network.trainable_params)  ] 
        qvalue_update = [ tf.assign(target, source)
                for target, source in zip(self.qvalue_network.trainable_params, source.qvalue_network.trainable_params)  ] 
        policy_update = [ tf.assign(target, source)
                for target, source in zip(self.policy_network.trainable_params, source.policy_network.trainable_params)  ] 
                
        sess.run(tf.variables_initializer(tf.global_variables(self.scope)))
                

        sess.run([state_update, qvalue_update, policy_update])