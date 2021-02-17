import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant import AntEnv
from networks.morl_learner import MORLLearner
from dummyenv import DummyEnv

env = AntEnv()
gamma = 0.96
horizon = 50
action_variance = 5
action_maximum = 1
running_samples = 200

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/morl2/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = MORLLearner(state_dim, action_dim, gamma=gamma, action_maximum=action_maximum)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

def is_done(state):
    notdone = np.isfinite(state).all() and state[0] >= 0.3 and state[0] <= 1.0
    return not notdone

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, "./data/trpo4/log_200.ckpt")


    mean_reward = 0.0
    learning_rate = 0.01

    for epoch in range(1, 5001):

        curreward = 0.
        actions = []
        mu = 0.
        sigma = 0.
        state = env.reset()
        for play in range(201):
            prevstate = state

            action = learner.get_action_optimal(state)
            state, reward, done, _ = env.step(action)

            actions.append(action)

            curreward += reward
            env.render()
            if(is_done(state)):
                break
        log_reward = curreward
        log_step = play
        print("Epoch " + str(epoch) + " Reward: " + str(curreward) + " Step:" + str(play))
        print("Mu " + str(np.mean(np.abs(actions), axis=0)) + " Std: " + str(np.std(actions, axis=0)))

    
        state_vector = []
        action_vector = []
        reward_vector = []
        nextstate_vector = []
        data_len = 0
        print("Epoch " + str(epoch) + " Start Collecting")
        while data_len < 2000:
            state = env.reset()
            for play in range(201):
                prevstate = state

                action = learner.get_action_collecting(state)
                state, reward, done, _ = env.step(action)

                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                reward_vector.append([reward])
                data_len += 1

                if(done):
                    break


        print("Epoch " + str(epoch) + " Start State Training")
        log_loss_state, log_loss_reward = learner.optimize_nextstate_batch(state_vector, action_vector, reward_vector, nextstate_vector)
        print("Epoch " + str(epoch) + " State Loss : " + str(log_loss_state) + " Reward Loss : " + str(log_loss_reward))
        print("Epoch " + str(epoch) + " Start Model Running")
        '''
        policy_state_vector = []
        policy_action_vector = []
        policy_adventage_vector = []
        for i in range(0, data_len, 100):
            cur_adventage_vector = []
            for j in range(action_variance):
                first_state = state_vector[i]
                first_action = learner.get_action_diverse(first_state)
                state, reward = learner.get_next(first_state, first_action)
                curreward = reward
                for k in range(horizon):
                    if is_done(state):
                        curreward -= 10.
                        break
                    state, reward = learner.get_next_with_policy(state)
                    curreward += reward * gamma ** (k + 1)

                policy_state_vector.append(first_state)
                policy_action_vector.append(first_action)
                cur_adventage_vector.append(curreward)

            cur_adventage_mean = np.mean(cur_adventage_vector)
            cur_adventage_std = np.std(cur_adventage_vector)

            for j in range(action_variance):
                policy_adventage_vector.append([(cur_adventage_vector[j] - cur_adventage_mean) / cur_adventage_std])
        '''
        dic = random.sample(range(data_len), running_samples)
        policy_state_vector = []
        policy_action_vector = []
        policy_adventage_vector = []
        first_state = [state_vector[i] for i in dic]
        adventage_sum_vector = np.zeros(running_samples)
        for i in range(action_variance):
            finished_vector = [False] * running_samples
            survive_vector = np.ones(running_samples)

            first_action = learner.get_actions_diverse(first_state)
            state, reward = learner.get_nexts(first_state, first_action)
            curreward = reward
            for j in range(horizon):
                survive_vector *= gamma
                for k in range(running_samples):
                    if is_done(state[k]) and not finished_vector[k]:
                        finished_vector[k] = True
                        survive_vector[k] = 0.
                state, reward = learner.get_nexts_with_policy(state)
                curreward = np.add(curreward, np.multiply(reward, survive_vector))
            policy_state_vector.extend(first_state)
            policy_action_vector.extend(first_action)
            policy_adventage_vector.extend(curreward)
            adventage_sum_vector = np.add(adventage_sum_vector, curreward)
        
        print(adventage_sum_vector)
        adventage_sum_vector /= action_variance
        for i in range(action_variance):
            policy_adventage_vector[running_samples * i : running_samples * (i+1)] = \
                np.subtract(policy_adventage_vector[running_samples * i : running_samples * (i+1)],  adventage_sum_vector)
        policy_adventage_vector = np.reshape(policy_adventage_vector, (-1, 1))


        print("Epoch " + str(epoch) + " Start Policy Training")
        overfitted = False
        while True:
            l, d = learner.optimize_policy_batch(learning_rate, policy_state_vector, policy_action_vector, policy_adventage_vector)
            print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
            if d > 1. or np.isnan(d):
                overfitted = True
                learning_rate /= 2.
            else:
                if overfitted:
                    break
                else:
                    learning_rate *= 2.
        learner.optimize_end()
        log_loss_policy = l
        print("Epoch " + str(epoch) + "Policy Training Loss : " + str(log_loss_policy))
            
                


        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_reward) +
            "\tSteps\t" + str(log_step) +
            "\tLossState\t" + str(log_loss_state) +
            "\tLossReward\t" + str(log_loss_reward) +
            "\tLossPolicy\t" + str(log_loss_policy) + "\n")


        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()