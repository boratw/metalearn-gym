
import numpy as np
import tensorflow as tf

from .mlp import MLP
from .stochastic_policy import StochasticPolicy


class MAMLLearner:
    def __init__(self, state_len, action_len, name = "", policy_lr=0.01, policy_batch = 16, meta_policy_lr = 0.01, policy_hidden_len=64, next_state_hidden_len=64, next_state_lr = 0.01, meta_next_state_lr = 0.01):
        with tf.variable_scope("MAMLLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_advantage = tf.placeholder(tf.float32, [None, 1], name="input_advantage")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_sigma = tf.placeholder(tf.float32, [], name="input_learning_rate")
            #self.input_kl = tf.placeholder(tf.float32, [], name="input_kl_div")

            self.next_state_network = MLP("nextstate", state_len, state_len, next_state_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.next_state_meta_network = MLP("next_state_meta", state_len, state_len, next_state_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            #self.policy_networks = [ StochasticPolicy("policy" + str(i), state_len, action_len, input_tensor=self.input_state,
            #    hidden_nonlinearity=tf.nn.tanh, input_sigma=self.input_sigma) for i in range(policy_batch) ]
            self.policy_network = StochasticPolicy("policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.tanh, input_sigma=self.input_sigma)
            self.new_policy_network = StochasticPolicy("new_policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.tanh, input_sigma=self.input_sigma) 
            self.poilcy_initialize = [ tf.assign(target, source) 
                for target, source in zip(self.new_policy_network.trainable_params, self.policy_network.trainable_params)  ]
            self.next_state_initialize = [ tf.assign(target, source) 
                for target, source in zip(self.next_state_network.trainable_params, self.next_state_meta_network.trainable_params)  ]

            input_advantage_var = tf.reduce_mean((self.input_advantage) ** 2)
            input_advantage_clipped = tf.clip_by_value(self.input_advantage / (tf.sqrt(input_advantage_var) + 0.1), -1, 2)
            self.new_policy_loss = tf.reduce_sum(((self.new_policy_network.mu - self.input_action) ** 2) * input_advantage_clipped)
            self.new_policy_update = tf.train.GradientDescentOptimizer(policy_lr).minimize(loss = self.new_policy_loss, var_list = self.new_policy_network.trainable_params) 

            #self.kl_div = tf.reduce_sum(tf.concat([ tf.reshape((a - b) ** 2, [-1]) for a , b in zip(self.new_policy_network.trainable_params, self.policy_network.trainable_params)], 0))

            self.policy_update = [ tf.assign(target, target + (source - target) * meta_policy_lr) 
                for target, source in zip(self.policy_network.trainable_params, self.new_policy_network.trainable_params)  ] 

            #self.policy_change = [ [ tf.assign(target, source) 
            #    for target, source in zip(self.policy_networks[0].trainable_params, self.policy_networks[i].trainable_params)  ] for i in range(policy_batch)  ]
            #self.policy_reserve = [ [ tf.assign(target, source) 
            #    for target, source in zip(self.policy_networks[i].trainable_params, self.policy_networks[0].trainable_params)  ] for i in range(policy_batch)  ]

            self.next_state_loss = tf.reduce_mean((self.next_state_network.layer_output - self.input_next_state) ** 2)
            self.next_state_train = tf.train.AdamOptimizer(next_state_lr).minimize(loss = self.next_state_loss, var_list = self.next_state_network.trainable_params)
            self.next_state_update = [ tf.assign(target, target + (source - target) * meta_next_state_lr) 
                for target, source in zip(self.next_state_meta_network.trainable_params, self.next_state_network.trainable_params)  ] 

            self.score = tf.reduce_mean(tf.exp(-tf.abs(self.next_state_network.layer_output - self.input_next_state) / 0.01))
            #self.policy_loss = tf.reduce_mean(tf.log((self.policy_network.mu - self.input_action) ** 2 + 1.) * self.input_advantage)
            #self.policy_loss = [ tf.reduce_mean(tf.log((self.policy_networks[i].mu - self.input_action) ** 2 + 1e-6) * self.input_advantage) for i in range(policy_batch)  ]



    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.new_policy_network.x, {self.input_state : np.array([input_state])})
        return output

    def get_action_deterministic(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.new_policy_network.mu, {self.input_state : np.array([input_state])})
        return output

    def get_action_stochastic(self, input_state, input_sigma):
        sess = tf.get_default_session()
        output = sess.run(self.new_policy_network.walk, {self.input_state : np.array([input_state]), self.input_sigma : input_sigma})
        return output

    def get_action_stochastic_batch(self, input_state, input_sigma):
        sess = tf.get_default_session()
        output = sess.run(self.new_policy_network.walk, {self.input_state : np.array(input_state), self.input_sigma : input_sigma})
        return output



    def optimize_policy_batch(self, input_state, input_action, input_advantage, input_next_state):

        sess = tf.get_default_session()
        _, loss, _ = sess.run([self.new_policy_update, self.new_policy_loss, self.next_state_train], {self.input_state : np.array(input_state),  self.input_action : np.array(input_action), 
            self.input_advantage : np.array(input_advantage), self.input_next_state : np.array(input_next_state)} )
        #div = sess.run( self.kl_div )
        return loss

    def get_score(self, input_state, input_action, input_next_state):
        sess = tf.get_default_session()
        score = sess.run(self.score, {self.input_state : np.array(input_state),  self.input_action : np.array(input_action), self.input_next_state : np.array(input_next_state)} )
        return score

    def init_policy(self):
        sess = tf.get_default_session()
        sess.run(self.poilcy_initialize)
        sess.run(self.next_state_initialize)

    def update_policy(self):
        sess = tf.get_default_session()
        sess.run(self.policy_update)
        sess.run(self.next_state_update)