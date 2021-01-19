import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  

from .networks.sac_learner import SACLearner

env = AntEnv()
state_dim = env.observation_space.flat_dim
action_dim = env.observation_space.flat_dim

LOG_DIR = "data/basic_train1/"
log_file = open(LOG_DIR + "log1.txt", "wt")

learner = SACLearner(state_dim, action_dim)

for epoch in range(10001):
    state = env.reset()
    reward_sum = 0
    cost_sum = [0., 0., 0.]
    for play in range(200):
        prev_state = state
        action = learner.get_action(state)
        state, reward, done, _ = env.step(action)

        reward_sum += reward
        costs = learner.optimize(prev_state, state, action, reward)
        print(cost)
        cost_sum.add(cost)

        env.render()
        if(done):
            break


    log_file.write("Episode\t" + str(epoch) + "\tStep\t" + str(play) +  "\tScore\t" + str(reward_sum) + "\tCost\t" + str(cost_sum) + "\n")
    if epoch % 5000 == 0:
        saver.save(sess, LOG_DIR + "log1_" + str(epoch) + ".ckpt")

        
env.close()