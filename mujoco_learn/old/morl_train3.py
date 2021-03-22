import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant6 import Ant6Env
from networks.morl_learner3 import MORLLearner
from dummyenv import DummyEnv

env = Ant6Env()
gamma = 0.96
horizon = 30
action_variance = 8
qvalue_variance = 4
action_maximum = 1.
running_samples = 200

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/morl9/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = MORLLearner(state_dim, action_dim, gamma=gamma, action_maximum=action_maximum, update_kl_div=1e-3)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] <= 1.0 and state[2] >= 0.27
    return not notdone

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, "./data/trpo4/log_200.ckpt")


    mean_reward = 0.0
    learning_rate = 0.01

    state_vector = []
    action_vector = []
    reward_vector = []
    nextstate_vector = []

    while len(state_vector) < 5000:
        state = env.reset()
        for play in range(201):
            prevstate = state

            action = learner.get_action_collecting(state)
            state, reward, done, _ = env.step(action * 0.5)

            state_vector.append(prevstate)
            action_vector.append(action)
            nextstate_vector.append(state)
            reward_vector.append([reward])

            if(done):
                break

    for epoch in range(1, 50001):

        data_len = 0

        mu = 0.
        sigma = 0.
        cur_state_vector = []
        cur_action_vector = []
        curreward = 0.
        state = env.reset()
        for play in range(201):
            prevstate = state

            action = learner.get_action_optimal(state)
            cur_state_vector.append(state)
            cur_action_vector.append(action)
            state, reward, done, _ = env.step(action * 0.5)

            state_vector.append(prevstate)
            action_vector.append(action)
            nextstate_vector.append(state)
            reward_vector.append([reward])
            data_len += 1

            curreward += reward
            env.render()
            if(is_done(state)):
                break
        log_reward = curreward
        log_step = play

        print("Epoch " + str(epoch) + " Reward: " + str(curreward) + " Step:" + str(play))

        print("Epoch " + str(epoch) + " Start Collecting")

        while data_len < 500:
            state = env.reset()
            for play in range(201):
                prevstate = state

                action = learner.get_action_optimal(state)
                cur_state_vector.append(state)
                cur_action_vector.append(action)
                state, reward, done, _ = env.step(action * 0.5)


                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                reward_vector.append([reward])
                data_len += 1

                if(is_done(state)):
                    break

        std_action = np.std(cur_action_vector, axis=0)
        print("Mu " + str(np.mean(np.abs(cur_action_vector), axis=0)) + " Std: " + str(std_action))
    
        while data_len < 1000:
            state = env.reset()
            for play in range(201):
                prevstate = state

                action = learner.get_action_collecting(state)
                state, reward, done, _ = env.step(action * 0.5)

                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                reward_vector.append([reward])
                data_len += 1

                if(done):
                    break

        dic = random.sample(range(len(state_vector)), 2000)
        state_vector_dic = [state_vector[i] for i in dic]
        action_vector_dic = [action_vector[i] for i in dic]
        reward_vector_dic = [reward_vector[i] for i in dic]
        nextstate_vector_dic = [nextstate_vector[i] for i in dic]

        print("Epoch " + str(epoch) + " Start State Training")
        log_loss_state, log_loss_reward = learner.optimize_nextstate_batch(state_vector_dic, action_vector_dic, reward_vector_dic, nextstate_vector_dic)
        print("Epoch " + str(epoch) + " State Loss : " + str(log_loss_state) + " Reward Loss : " + str(log_loss_reward))
        print("Epoch " + str(epoch) + " Start Model Running")

        dic = random.sample(range(len(state_vector)), running_samples)
        policy_state_vector = []
        policy_action_vector = []
        policy_qvalue_vector = []
        #policy_adventage_vector = []
        first_state = [state_vector[i] for i in dic]
        qvalue_sum_vector = np.zeros(running_samples)
        #qvalue_square_sum_vector = np.zeros(running_samples)
        for i in range(action_variance):

            first_action = learner.get_actions_diverse(first_state)
            #currewards = []
            #for j in range(qvalue_variance):

            finished_vector = [False] * running_samples
            survive_vector = np.ones(running_samples) * gamma
            state, reward = learner.get_nexts(first_state, first_action)
            curreward = reward
            for j in range(horizon):
                for k in range(running_samples):
                    if is_done(state[k]) and not finished_vector[k]:
                        finished_vector[k] = True
                        survive_vector[k] = 0.
                        curreward[k] -= 1. * gamma ** j
                state, reward = learner.get_nexts_with_policy_optimal(state)
                curreward = np.add(curreward, np.multiply(reward, survive_vector))
                survive_vector *= gamma
            #qvalue = learner.get_qvalues_with_policy(state)
            #curreward = np.add(curreward, np.multiply(qvalue, survive_vector * 0.5))
            #currewards.append(curreward)

            #curreward = curreward - np.sum(np.square(first_action), axis=1) * 0.01
            #maxreward = np.amax(currewards, axis=0)
            
            policy_state_vector.extend(first_state)
            policy_action_vector.extend(first_action)
            policy_qvalue_vector.extend(curreward)
            #policy_adventage_vector.extend(qvalue)
            qvalue_sum_vector = np.add(qvalue_sum_vector, curreward)
            #qvalue_square_sum_vector = np.add(qvalue_square_sum_vector, np.multiply(curreward, curreward) )
        
        mean_qvalue = np.mean(policy_qvalue_vector)

        qvalue_sum_vector /= action_variance
        for i in range(action_variance):
            policy_qvalue_vector[running_samples * i : running_samples * (i+1)] = \
                np.subtract(policy_qvalue_vector[running_samples * i : running_samples * (i+1)],  qvalue_sum_vector)

        policy_qvalue_vector = np.reshape(policy_qvalue_vector, (-1, 1))
        print("Epoch " + str(epoch) + " Mean Qvalue : " + str(mean_qvalue))
        
        print("Epoch " + str(epoch) + " Start Qvalue Training")
        log_loss_qvalue = learner.optimize_qvalue_batch(policy_state_vector, policy_action_vector, policy_qvalue_vector)
        #log_loss_qvalue = learner.optimize_qvalue_batch(policy_state_vector)
        print("Epoch " + str(epoch) + " Qvalue Training Loss : " + str(log_loss_qvalue))
        
        print("Epoch " + str(epoch) + " Start Policy Training")
        log_loss_policy = learner.optimize_policy_batch(cur_state_vector)
        print("Epoch " + str(epoch) + " Policy Training Loss : " + str(log_loss_policy))
            
                


        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_reward) +
            "\tSteps\t" + str(log_step) +
            "\tMeanQValue\t" + str(mean_qvalue) +
            "\tLossState\t" + str(log_loss_state) +
            "\tLossReward\t" + str(log_loss_reward) +
            "\tLossQvalue\t" + str(log_loss_qvalue) +
            "\tLossPolicy\t" + str(log_loss_policy) + "\n")


        cut = int(len(state_vector) * 0.05)
        state_vector = state_vector[cut:]
        action_vector = action_vector[cut:]
        reward_vector = reward_vector[cut:]
        nextstate_vector = nextstate_vector[cut:]

        if epoch % 1000 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()
