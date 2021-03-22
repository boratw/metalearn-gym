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
horizon = 10
action_variance = 20
action_maximum = 1
running_samples = 200

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/morl4/"
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

    state_vector = []
    action_vector = []
    reward_vector = []
    nextstate_vector = []

    for epoch in range(1, 5001):

        data_len = 0

        curreward = 0.
        state = env.reset()
        for play in range(201):
            prevstate = state

            action = learner.get_action_optimal(state)
            state, reward, done, _ = env.step(action)

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

        mu = 0.
        sigma = 0.
        actions = []
        action_data_len = 0
        while action_data_len < 500:
            state = env.reset()
            for play in range(101):
                prevstate = state

                action = learner.get_action_optimal(state)
                actions.append(action)
                action = learner.get_action_collecting(state)
                state, reward, done, _ = env.step(action)


                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                reward_vector.append([reward])
                data_len += 1
                action_data_len += 1

                if(is_done(state)):
                    break

        std_action = np.std(actions, axis=0)
        print("Mu " + str(np.mean(np.abs(actions), axis=0)) + " Std: " + str(std_action))
    
        while data_len < 1000:
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

        dic = random.sample(range(len(state_vector)), 1000)
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
        policy_adventage_vector = []
        first_state = [state_vector[i] for i in dic]
        adventage_sum_vector = np.zeros(running_samples)
        adventage_square_sum_vector = np.zeros(running_samples)
        for i in range(action_variance):

            first_action = learner.get_actions_diverse(first_state)
            qvalue = learner.get_qvalues(first_state, first_action)
            
            finished_vector = [False] * running_samples
            survive_vector = np.ones(running_samples) * gamma
            state, reward = learner.get_nexts(first_state, first_action)
            curreward_stochastic = reward
            for j in range(horizon):
                for k in range(running_samples):
                    if is_done(state[k]) and not finished_vector[k]:
                        finished_vector[k] = True
                        survive_vector[k] = 0.
                        #curreward_optimal[k] -= 10. * gamma ** j
                state, reward = learner.get_nexts_sto_with_policy(state)
                curreward_stochastic = np.add(curreward_stochastic, np.multiply(reward, survive_vector))
                survive_vector *= gamma
            qvalue = learner.get_qvalues_with_policy(state)
            curreward_stochastic = np.add(curreward_stochastic, np.multiply(qvalue, survive_vector * 0.5))
            
            '''
            finished_vector = [False] * running_samples
            survive_vector = np.ones(running_samples) * gamma
            state, reward = learner.get_nexts(first_state, first_action)
            curreward_stochastic2 = reward
            for j in range(horizon):
                for k in range(running_samples):
                    if is_done(state[k]) and not finished_vector[k]:
                        finished_vector[k] = True
                        survive_vector[k] = 0.
                        #curreward_stochastic[k] -= 10. * gamma ** j
                state, reward = learner.get_nexts_sto2_with_policy(state)
                curreward_stochastic2 = np.add(curreward_stochastic2, np.multiply(reward, survive_vector))
                survive_vector *= gamma
            qvalue = learner.get_qvalues_with_policy(state)
            curreward_stochastic2 = np.add(curreward_stochastic2, np.multiply(qvalue, survive_vector * 0.5))

            curreward = np.max([curreward_stochastic, curreward_stochastic2], axis=0)
            '''
            curreward = curreward - np.sum(np.square(first_action), axis=1) * 0.01

            
            policy_state_vector.extend(first_state)
            policy_action_vector.extend(first_action)
            policy_qvalue_vector.extend(curreward)
            policy_adventage_vector.extend(qvalue)
            adventage_sum_vector = np.add(adventage_sum_vector, qvalue)
            adventage_square_sum_vector = np.add(adventage_square_sum_vector, np.multiply(qvalue, qvalue) )
        
        policy_qvalue_vector = np.reshape(policy_qvalue_vector, (-1, 1))

        adventage_sum_vector /= action_variance
        adventage_square_sum_vector /= action_variance
        adventage_std_vector = np.sqrt(adventage_square_sum_vector - np.multiply(adventage_sum_vector, adventage_sum_vector)) + 1e-8
        for i in range(action_variance):
            policy_adventage_vector[running_samples * i : running_samples * (i+1)] = \
                np.divide(np.subtract(policy_adventage_vector[running_samples * i : running_samples * (i+1)],  adventage_sum_vector), adventage_std_vector)
        policy_adventage_vector = np.reshape(policy_adventage_vector, (-1, 1))


        
        print("Epoch " + str(epoch) + " Start Qvalue Training")
        log_loss_qvalue = learner.optimize_qvalue_batch(policy_state_vector, policy_action_vector, policy_qvalue_vector)
        #log_loss_qvalue = learner.optimize_qvalue_batch(policy_state_vector)
        print("Epoch " + str(epoch) + " Qvalue Training Loss : " + str(log_loss_qvalue))
        
        print("Epoch " + str(epoch) + " Start Policy Training")
        log_loss_policy = learner.optimize_policy_batch(std_action, policy_state_vector, policy_action_vector, policy_adventage_vector)
        print("Epoch " + str(epoch) + " Policy Training Loss : " + str(log_loss_policy))
            
                


        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_reward) +
            "\tSteps\t" + str(log_step) +
            "\tStdAction\t" + str(std_action) +
            "\tLossState\t" + str(log_loss_state) +
            "\tLossReward\t" + str(log_loss_reward) +
            "\tLossQvalue\t" + str(log_loss_qvalue) +
            "\tLossPolicy\t" + str(log_loss_policy) + "\n")


        cut = int(len(state_vector) * 0.95)
        state_vector = state_vector[cut:]
        action_vector = action_vector[cut:]
        reward_vector = reward_vector[cut:]
        nextstate_vector = nextstate_vector[cut:]

        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()