
import numpy as np
import tensorflow as tf

from .mlp import MLP




class ModelEnv:

    def __init__(self, state_len, action_len, survive_fn, hidden_len=256, lr=0.001):
        self.survive_fn = survive_fn
        self.state_len = state_len
        with tf.variable_scope("FakeEnv"): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")


            self.network = MLP("network", state_len, state_len + 1, hidden_len, input_tensor=self.input_state,
                hidden_nonlinearity=tf.nn.leaky_relu, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)

            self.loss = tf.reduce_mean((tf.concat([self.input_next_state, self.input_reward], 1)  - self.network.layer_output) ** 2)
            
            self.train = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list = self.network.trainable_params)

    def get_batch(self, input_state, input_action):

        sess = tf.get_default_session()
        output = sess.run(self.network.layer_output, {self.input_state : np.array(input_state), self.input_action : np.array(input_action)})

        state = output[:, :self.state_len]
        reward = np.expand_dims(output[:, -1], axis=1)
        survive = self.get_survive(state)

        return state, reward, survive

    def optimize(self, input_state, input_next_state, input_action, input_reward):
        input_list = {self.input_state : np.array(input_state), self.input_next_state : np.array(input_next_state), 
            self.input_action : np.array(input_action), self.input_reward : np.array(input_reward)}

        
        sess = tf.get_default_session()
        loss, _ = sess.run([self.loss, self.train], input_list)

        return loss

    def get_survive(self, state):
        return [ [1.0 if self.survive_fn(x) else 0.0] for x in state ]