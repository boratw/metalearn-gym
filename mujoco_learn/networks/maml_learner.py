
import numpy as np
import tensorflow as tf

from .mlp import MLP
from .stochastic_policy import StochasticPolicy


class MAMLLearner:
    def __init__(self, state_len, action_len, name="", policy_lr=0.01, policy_hidden_len=64, policy_meta_lr=0.1, value_lr=0.01, value_hidden_len=128, nextstate_lr=0.01, nextstate_hidden_len=128, value_gamma=0.96, value_meta_lr=0.1):
        with tf.variable_scope("MAMLLearner" + name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_value = tf.placeholder(tf.float32, [None, 1], name="input_value")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_update_ratio = tf.placeholder(tf.float32, [], name="input_update_ratio")
            self.input_learning_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")
            self.input_sigma_rate = tf.placeholder(tf.float32, [], name="input_sigma_rate")

            self.value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.input_next_state)
            self.nextstate_network = MLP("nextstate", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.policy_network = StochasticPolicy("policy", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, input_sigma_rate=self.input_sigma_rate)

            self.value_meta_network = MLP("value_meta", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.input_next_state)
            self.nextstate_meta_network = MLP("nextstate_meta", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.policy_meta_network = StochasticPolicy("policy_meta", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, input_sigma_rate=self.input_sigma_rate)

            self.value_grad_store_network = MLP("value_grad", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.input_next_state)
            self.nextstate_grad_store_network = MLP("nextstate_grad", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
            self.policy_grad_store_network = StochasticPolicy("policy_grad", state_len, action_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, input_sigma_rate=self.input_sigma_rate) 
            self.value_grad_init = [ tf.assign(target, tf.zeros(target.shape)) for target in self.value_grad_store_network.trainable_params ]
            self.nextstate_grad_init = [ tf.assign(target, tf.zeros(target.shape)) for target in self.nextstate_grad_store_network.trainable_params ]
            self.policy_grad_init = [ tf.assign(target, tf.zeros(target.shape)) for target in self.policy_grad_store_network.trainable_params ]

            self.value_train_init = [ tf.assign(target, source) for target, source in zip(self.value_network.trainable_params, self.value_meta_network.trainable_params)]
            self.nextstate_train_init = [ tf.assign(target, source) for target, source in zip(self.nextstate_network.trainable_params, self.nextstate_meta_network.trainable_params)]
            self.policy_train_init = [ tf.assign(target, source) for target, source in zip(self.policy_network.trainable_params, self.policy_meta_network.trainable_params)]

            self.value_loss = tf.reduce_mean((self.value_network.layer_output - self.input_value) ** 2)
            self.value_train = tf.train.AdamOptimizer(value_lr).minimize(loss = self.value_loss, var_list = self.value_network.trainable_params)
            self.nextstate_loss = tf.reduce_mean((self.nextstate_network.layer_output - self.input_next_state) ** 2)
            self.nextstate_train = tf.train.AdamOptimizer(nextstate_lr).minimize(loss = self.nextstate_loss, var_list = self.nextstate_network.trainable_params)

            self.next_average_state_network = MLP("nextstate", state_len, state_len, nextstate_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy_meta_network.mu, reuse=True)
            self.next_average_value_network = MLP("value", state_len, 1, value_hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.next_average_state_network.layer_output, reuse=True)
            next_value = tf.stop_gradient(self.next_average_value_network.layer_output)
            
            self.policy_loss = tf.reduce_mean( -(self.input_value - next_value) * self.policy_network.log_prob(self.input_action) )
            self.meta_policy_loss = tf.reduce_mean( -(self.input_value - next_value) * self.policy_meta_network.log_prob(self.input_action) )
            self.policy_train = tf.train.GradientDescentOptimizer(self.input_learning_rate).minimize(loss = self.policy_loss + self.policy_network.regularization_loss, var_list = self.policy_network.trainable_params) 

            self.policy_kl_div = self.get_kl_div()
            self.loss_diff = tf.reduce_sum(self.meta_policy_loss - self.policy_loss)
            loss_update = tf.exp(tf.clip_by_value(self.input_update_ratio, -2, 2))

            self.value_store = [ tf.assign(target, target + (source - origin)) 
                for target, source, origin in zip(self.value_grad_store_network.trainable_params, self.value_network.trainable_params, self.value_meta_network.trainable_params)  ] 
            self.nextstate_store = [ tf.assign(target, target + (source - origin)) 
                for target, source, origin in zip(self.nextstate_grad_store_network.trainable_params, self.nextstate_network.trainable_params, self.nextstate_meta_network.trainable_params)  ] 
            self.policy_store = [ tf.assign(target, target + (source - origin) * loss_update) 
                for target, source, origin in zip(self.policy_grad_store_network.trainable_params, self.policy_network.trainable_params, self.policy_meta_network.trainable_params)  ] 

            self.value_update = [ tf.assign(target, target + source * value_meta_lr) 
                for target, source in zip(self.value_meta_network.trainable_params, self.value_grad_store_network.trainable_params)  ] 
            self.nextstate_update = [ tf.assign(target, target + source * value_meta_lr) 
                for target, source in zip(self.nextstate_meta_network.trainable_params, self.nextstate_grad_store_network.trainable_params)  ] 
            self.policy_update = [ tf.assign(target, target + source * policy_meta_lr) 
                for target, source in zip(self.policy_meta_network.trainable_params, self.policy_grad_store_network.trainable_params)  ] 

    def get_kl_div(self):
        numerator = tf.square(self.policy_meta_network.mu - self.policy_network.mu) + tf.square(self.policy_meta_network.std) - tf.square(self.policy_network.std)
        denominator = 2 * tf.square(self.policy_network.std) + 1e-8
        return tf.reduce_sum( numerator / denominator + self.policy_network.logsig - self.policy_meta_network.logsig)


    def get_action(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_meta_network.x, {self.input_state : np.array([input_state])})
        return output

    def get_action_deterministic(self, input_state, input_sigma_rate = 1.0):
        sess = tf.get_default_session()
        output = sess.run(self.policy_meta_network.mu, {self.input_state : np.array([input_state])})
        return output

    def get_action_stochastic(self, input_state, input_sigma_rate = 1.0):
        sess = tf.get_default_session()
        output, m, s = sess.run([self.policy_meta_network.walk,self.policy_meta_network.mu, self.policy_meta_network.logsig], {self.input_state : np.array([input_state]), self.input_sigma_rate : input_sigma_rate})
        return output, m, s

    def get_action_stochastic_batch(self, input_state):
        sess = tf.get_default_session()
        output = sess.run(self.policy_meta_network.walk, {self.input_state : np.array(input_state), self.input_sigma_rate : input_sigma_rate})
        return output

    def gradient_init(self):
        sess = tf.get_default_session()
        sess.run([self.value_grad_init, self.nextstate_grad_init, self.policy_grad_init])

    def optimize_start(self):
        sess = tf.get_default_session()
        sess.run([self.value_train_init, self.nextstate_train_init])


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
        sess.run(self.policy_train_init)
        _, loss = sess.run([self.policy_train, self.policy_loss], {self.input_learning_rate:input_learning_rate, self.input_state : np.array(input_state),  self.input_action : np.array(input_action),
            self.input_value : np.array(input_value), self.input_next_state : np.array(input_next_state)} )
        div, self.cur_diff = sess.run([self.policy_kl_div, self.loss_diff], {self.input_learning_rate:input_learning_rate, self.input_state : np.array(input_state),  self.input_action : np.array(input_action),
            self.input_value : np.array(input_value), self.input_next_state : np.array(input_next_state)})

        return loss, div, self.cur_diff
    
    def optimize_end(self):
        sess = tf.get_default_session()
        sess.run([self.value_store, self.nextstate_store, self.policy_store], {self.input_update_ratio : self.cur_diff})
    
    def gradient_assign(self):
        sess = tf.get_default_session()
        sess.run([self.value_update, self.nextstate_update, self.policy_update])
