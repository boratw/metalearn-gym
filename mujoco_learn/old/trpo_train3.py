import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant import AntEnv
from networks.trpo_learner import TRPOLearner
from dummyenv import DummyEnv


env = AntEnv()
gamma = 0.96

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/trpo3/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = TRPOLearner(state_dim, action_dim, value_gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)


    mean_reward = 0.0

    for epoch in range(1, 5001):
        log_rewards = 0.
        log_steps = 0.
        log_loss_policy = 0.
        log_loss_value = 0.
        log_divergences = 0.

        state_vector = []
        action_vector = []
        reward_vector = []
        value_vector = []
        nextstate_vector = []

        for batch in range(8):
            batchstart = len(state_vector)
            state = env.reset()
            curreward = 0.
            reward = 0.

            for play in range(201):
                prevstate = state

                action = learner.get_action_stochastic(state)[0]
                state, reward, done, _ = env.step(action)

                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                value_vector.append([reward])
                reward_vector.append([reward])

                curreward += reward
                if batch == 0:
                    env.render()
                if(done):
                    break

            #if play == 200:
            #    value_vector[-1][0] +=  (np.mean(reward_vector[-10]) / 10.) * (1. / (1. - gamma) - 1.)
            #else:
            if play != 200:
                reward_vector[-1][0] -= 10.0
                value_vector[-1][0] -= 10.0

            for i in range(len(value_vector) - 1, batchstart, -1):
                value_vector[i-1][0] =  gamma * value_vector[i][0] + value_vector[i-1][0]

            if play > 150:
                cut = play - 150
                state_vector = state_vector[:-cut]
                action_vector = action_vector[:-cut]
                nextstate_vector = nextstate_vector[:-cut]
                value_vector = value_vector[:-cut]
                reward_vector = reward_vector[:-cut]

            log_rewards += curreward
            log_steps += play
            print("Epoch " + str(epoch) + " Reward: " + str(curreward) + " Step:" + str(play))

        log_rewards /= 8.
        log_steps /= 8.

        overfitted = False
        learning_rate = 0.01
        while True:
            dic = random.sample(range(len(state_vector)), 256 if len(state_vector) >= 256 else len(state_vector))
            state_vector_dic = [state_vector[x] for x in dic]
            action_vector_dic = [action_vector[x] for x in dic]
            reward_vector_dic = [reward_vector[x] for x in dic]
            nextstate_vector_dic = [nextstate_vector[x] for x in dic]

            l, d = learner.optimize_policy_batch(learning_rate, state_vector_dic, action_vector_dic, reward_vector_dic, nextstate_vector_dic)
            
            print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
            if d > 0.01:
                overfitted = True
                learning_rate /= 2.
            else:
                if overfitted:
                    break
                else:
                    learning_rate *= 2.
        learner.optimize_end()

        
        log_loss_policy = l
        log_divergences = d

        print("Loss " + str(log_loss_policy))
        print("Divergence " + str(log_divergences))

        for i in range(32):
            dic = random.sample(range(len(state_vector)), 64 if len(state_vector) >= 64 else len(state_vector))
            state_vector_dic = [state_vector[x] for x in dic]
            value_vector_dic = [value_vector[x] for x in dic]

            v = learner.optimize_value_batch(state_vector_dic, value_vector_dic)
            log_loss_value += v
        log_loss_value /= 32
        print("Value loss " + str(log_loss_value))

        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_rewards) +
            "\tSteps\t" + str(log_steps) +
            "\tLossValue\t" + str(log_loss_value) +
            "\tLossPolicy\t" + str(log_loss_policy) +
            "\tDivergence\t" + str(log_divergences ) + "\n")


        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()