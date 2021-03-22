import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant_randvel import AntRandVelEnv
from gym.envs.mujoco.ant import AntEnv
from networks.maml_learner import MAMLLearner
from dummyenv import DummyEnv

env = AntEnv()

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/trpo/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = MAMLLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    saver.restore(sess, "./data/trpo/log_3000.ckpt")

    mean_reward = 0.01
    for epoch in range(1, 101):
        state = env.reset()
        for play in range(201):
            action = learner.get_action_deterministic(state)[0]
            state, reward, done, _ = env.step(action)
            env.render()
            if(done):
                break
env.close()