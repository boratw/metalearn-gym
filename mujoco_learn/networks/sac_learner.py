
import numpy as np
import tensorflow as tf

from .mlp import MLP


class SACLearner:
    def __init__(self, state_len, action_len, 
        value_hidden_len=32, qvalue_hidden_len=32, policy_hidden_len=32,
        train_sampling=16, inference_sampling=32, gamma=0.95):

        self.action_len = action_len
        self.train_sampling = train_sampling
        self.inference_sampling = inference_sampling
        self.gamma = gamma

        with tf.variable_scope("SACLearner"): 
            self.value_network = MLP("value", state_len, 1, value_hidden_len)
            self.value_input = self.value_network.layer_input
            self.value_output = self.value_network.layer_output

            self.tvalue_network = MLP("tvalue", state_len, 1, value_hidden_len)
            self.tvalue_input = self.value_network.layer_input
            self.tvalue_output = self.value_network.layer_output

            self.qvalue_network = MLP("qvalue", state_len + action_len, 1, qvalue_hidden_len)
            self.qvalue_input = self.qvalue_network.layer_input
            self.qvalue_output = self.qvalue_network.layer_output

            self.policy_network = MLP("policy", state_len + action_len, 1, policy_hidden_len, output_decorate=tf.log)
            self.policy_input = self.policy_network.layer_input
            self.policy_output = self.policy_network.layer_output

            


    def get_action(self, input_state):
        noise_batch = np.random.standard_normal( (self.train_sampling, self.action_len) )
        sample_batch = np.hstack((np.tile(input_state, (self.train_sampling, 1)), noise_batch))
        output = self.policy_network.get_output(sample_batch)

        maxarg = np.argmax(output)
        return noise_batch[maxarg]

    def optimize(self, input_state, input_next_state, input_action, input_reward):
        noise_batch = np.random.standard_normal( (self.inference_sampling, self.action_len) )
        state_batch = np.tile(input_state, (self.inference_sampling, 1))
        state_noise_batch = np.hstack((state_batch, noise_batch))

        qvalue = self.qvalue_network.get_output(state_noise_batch)
        qvalue_average = np.mean(qvalue)
        policy = self.policy_network.get_output(state_noise_batch)
        
        value_cost = self.value_network.optimize(state_batch, numpy.subtract(qvalue - policy))
        policy_cost = self.policy_network.optimize(state_batch, qvalue - qvalue_average) 

        state_action = np.array([np.concatenate(input_state, input_action)])
        value = self.tvalue_network.get_output(np.array([input_next_state]))
        qvalue_cost = self.qvalue_network.optimize(state_action, np.array([[value[0][0] * self.gamma + input_reward]]))


        return [value_cost, policy_cost, qvalue_cost]
