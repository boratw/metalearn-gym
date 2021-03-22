
import numpy as np
import tensorflow as tf

from .mlp import MLP
from .gaussian_policy import GaussianPolicy


class SACLearner:
    def __init__(self, state_len, action_len, name="", 
        state_hidden_len=256, value_hidden_len=256, qvalue_hidden_len=256, policy_hidden_len=256,
        state_lr=0.001, value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=0.96):

        self.scope = "SACLearner" + name

        with tf.variable_scope(self.scope): 
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_survive = tf.placeholder(tf.float32, [None, 1], name="input_survive")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")

            self.state_network = MLP("state", state_len, state_len, state_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)

            self.value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu)
            self.value_target_network = MLP("tvalue", state_len, 1, value_hidden_len, input_tensor=self.state_network.layer_output,
                hidden_nonlinearity=tf.nn.leaky_relu)

            self.qvalue1_network = MLP("qvalue1", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)

            self.qvalue2_network = MLP("qvalue2", state_len, 1, qvalue_hidden_len,  hidden_nonlinearity=tf.nn.leaky_relu,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)

            self.policy_network = GaussianPolicy("policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.tanh)

            self.qvalue1_network_forpolicy = MLP("qvalue1", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, reuse=True,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len,
                additional_input_tensor=self.policy_network.squashed_x)

            self.qvalue2_network_forpolicy = MLP("qvalue2", state_len, 1, qvalue_hidden_len, hidden_nonlinearity=tf.nn.leaky_relu, reuse=True,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len,
                additional_input_tensor=self.policy_network.squashed_x)

            #self.policy_prior = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros(action_len), scale_diag=tf.ones(action_len))
            #self.policy_prior_log_probs = self.policy_prior.log_prob(self.policy_network.squashed_x)

            self.state_loss = tf.reduce_mean((self.input_next_state - self.state_network.layer_output) ** 2)
            self.state_train = tf.train.AdamOptimizer(state_lr).minimize(self.state_loss,
                var_list = self.state_network.trainable_params)

            self.value_assign = self.value_target_network.build_add_weighted(self.value_network, 0.1)


            soft_value = self.input_reward + self.input_survive * tf.stop_gradient(self.value_target_network.layer_output) * gamma
            self.qvalue1_loss = tf.reduce_mean((soft_value - self.qvalue1_network.layer_output) ** 2)
            self.qvalue1_train = tf.train.AdamOptimizer(qvalue_lr).minimize(self.qvalue1_loss,
                var_list = self.qvalue1_network.trainable_params)
            self.qvalue2_loss = tf.reduce_mean((soft_value - self.qvalue2_network.layer_output) ** 2)
            self.qvalue2_train = tf.train.AdamOptimizer(qvalue_lr).minimize(self.qvalue2_loss,
                var_list = self.qvalue2_network.trainable_params)

            #self.policy_loss = tf.reduce_mean((self.policy_network.log_pi * tf.stop_gradient(self.policy_network.log_pi - self.qvalue1_network_forpolicy.layer_output + self.value_network.layer_output - self.policy_prior_log_probs)) )
            #self.policy_loss = tf.reduce_mean((self.policy_network.log_pi * tf.stop_gradient(self.qvalue1_network_forpolicy.layer_output + self.policy_prior_log_probs - self.value_network.layer_output - self.policy_network.log_pi)) )
            self.policy_loss = tf.reduce_mean(self.policy_network.log_pi - self.qvalue1_network_forpolicy.layer_output)
            self.policy_train = tf.train.AdamOptimizer(policy_lr).minimize(loss = (self.policy_loss + self.policy_network.regularization_loss),
                var_list = self.policy_network.trainable_params)
            self.policy_likelihood = self.policy_network.log_li(self.input_action)
            
            min_log_target = tf.minimum(self.qvalue1_network_forpolicy.layer_output, self.qvalue2_network_forpolicy.layer_output)
            self.value_loss = tf.reduce_mean((self.value_network.layer_output - tf.stop_gradient(min_log_target - self.policy_network.log_pi)) ** 2) 
            self.value_train = tf.train.AdamOptimizer(value_lr).minimize(loss = self.value_loss,
                var_list = self.value_network.trainable_params)

            diff_input_value = self.input_value - tf.reduce_mean(self.input_value, keepdims=True)
            diff_output_value = self.value_network.layer_output - tf.reduce_mean(self.value_network.layer_output, keepdims=True)
            self.value_corr = tf.reduce_mean( diff_input_value * diff_output_value) / (tf.sqrt(tf.reduce_mean(diff_input_value * diff_input_value)) * tf.sqrt(tf.reduce_mean(diff_output_value * diff_output_value)) )


    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.squashed_x, {self.input_state : np.array([input_state])})
        return output

    def get_action_deterministic(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.squashed_mu, {self.input_state : np.array([input_state])})
        return output

    def get_action_stochastic(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.squashed_walk, {self.input_state : np.array([input_state])})
        return output

    def get_action_stochastic_batch(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_network.squashed_walk, {self.input_state : np.array(input_state)})
        return output

    def get_next_state(self, input_state, input_action):
        sess = tf.get_default_session()
        output = sess.run(self.state_network.layer_output, {self.input_state : np.array([input_state]), self.input_action : np.array([input_action])})
        return output[0]


    def get_value_correlation(self, input_state, input_value):
        sess = tf.get_default_session()
        output = sess.run(self.value_corr, {self.input_state : np.array(input_state), self.input_value : np.array(input_value)})
        return output

    def get_values(self, input_state, input_action):
        sess = tf.get_default_session()
        q, v, p = sess.run([self.qvalue1_network.layer_output, self.value_network.layer_output, self.policy_likelihood],
            {self.input_state : np.array([input_state]), self.input_action : np.array([input_action])})
        return q[0][0], v[0][0], p[0]

    def optimize(self, input_state, input_next_state, input_action, input_reward, input_survive):
        input_list = {self.input_state : np.array([input_state]), self.input_next_state : np.array([input_next_state]), 
            self.input_action : np.array([input_action]), self.input_reward : np.array([input_reward]), self.input_survive : np.array([input_survive])}

        sess = tf.get_default_session()
        q, v, p = sess.run([self.qvalue1_network.layer_output, self.value_target_network.layer_output, self.policy_network.log_pi], input_list)
        ql, vl, pl, sl, _, _, _, _, _ = sess.run([self.qvalue1_loss, self.value_loss, self.policy_loss, self.state_loss,
            self.qvalue1_train, self.qvalue2_train,  self.policy_train, self.value_train, self.state_train], input_list)

        return q, v, p, ql, vl, pl, sl

    def optimize_batch(self, input_state, input_next_state, input_action, input_reward, input_survive):
        input_list = {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state), 
            self.input_action : np.array(input_action), self.input_reward : np.array(input_reward), self.input_survive : np.array(input_survive)}

        sess = tf.get_default_session()
        q, v, p = sess.run([self.qvalue1_network.layer_output, self.value_target_network.layer_output, self.policy_network.log_pi], input_list)
        ql, vl, pl, sl, _, _, _, _, _ = sess.run([self.qvalue1_loss, self.value_loss, self.policy_loss, self.state_loss,
            self.qvalue1_train, self.qvalue2_train,  self.policy_train, self.value_train, self.state_train], input_list)

        return q, v, p, ql, vl, pl, sl

    def value_network_initialize(self):
        value_init = self.value_target_network.build_add_weighted(self.value_network, 1.0)

        sess = tf.get_default_session()
        sess.run(value_init)


    def value_network_update(self):
        sess = tf.get_default_session()
        sess.run(self.value_assign)

    def init_from_other(self, source):
        sess = tf.get_default_session()
        state_update = [ tf.assign(target, source)
                for target, source in zip(self.state_network.trainable_params, source.state_network.trainable_params)  ] 
        value_update = [ tf.assign(target, source)
                for target, source in zip(self.value_network.trainable_params, source.value_network.trainable_params)  ] 
        qvalue1_update = [ tf.assign(target, source)
                for target, source in zip(self.qvalue1_network.trainable_params, source.qvalue1_network.trainable_params)  ] 
        qvalue2_update = [ tf.assign(target, source)
                for target, source in zip(self.qvalue2_network.trainable_params, source.qvalue2_network.trainable_params)  ] 
        policy_update = [ tf.assign(target, source)
                for target, source in zip(self.policy_network.trainable_params, source.policy_network.trainable_params)  ] 
                
        sess.run(tf.variables_initializer(tf.global_variables(self.scope)))
        sess.run([state_update, value_update, qvalue1_update, qvalue2_update, policy_update])