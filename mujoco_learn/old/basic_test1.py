import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.ant6 import Ant6Env

from networks.sac_learner import SACLearner

env = Ant6Env()
state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

learner = SACLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    saver.restore(sess, "./data/basic_moml/log_300.ckpt")

    for epoch in range(1, 20):
        state = env.reset()
        for play in range(200):
            prev_state = state
            action = learner.get_action_stochastic(state)
            state, reward, done, _ = env.step(action)

            if(done):
                break
            env.render()
            
env.close()