import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant import AntEnv
import xml.etree.ElementTree as elemTree

from networks.trpo_learner import TRPOLearner

env1 = AntEnv()
state_dim = env1.get_current_obs().size
action_dim = env1.action_space.shape[0]
gamma = 0.96

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/trpo_train3/"
log_file = open(LOG_DIR + "log.txt", "wt")

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] <= 1.0 and state[2] >= 0.27
    return not notdone

learner1 = TRPOLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

log_file.write("Episode\t" +
    "Env1_Score\tEnv1_Output_Mu\tEnv1_Output_Std\tEnv1_Value\tEnv1_Policy_old\tEnv1_Policy_new\tEnv1_Policy_stepsize\n")

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    vector_len1 = 0
    state_vector1 = []
    action_vector1 = []
    advantage_vector1 = []
    value_vector1 = []
    old_mu_vector1 = []
    old_std_vector1 = []

    for epoch in range(100):
        state = env1.reset()
        reward_store = []
        for play in range(200):
            prev_state = state
            mu, std = learner1.get_action_stochastic(state)
            action = np.clip(mu + np.exp(std) * np.random.randn(action_dim), -1.0, 1.0)
            state, reward, done, _ = env1.step(action)
            if(done):
                reward -= 10.
            state_vector1.append(prev_state)
            action_vector1.append(action)
            old_mu_vector1.append(mu)
            old_std_vector1.append(std)
            reward_store.append(reward)

            vector_len1 += 1
            if(done):
                break

        reward_store = np.array(reward_store)
        value_store = np.array(reward_store)
        for backprop in range(1, (30 if play > 31 else play - 1)):
            value_store[:-backprop] += reward_store[backprop:] * (gamma ** backprop)
        for i in range(len(reward_store)):
            advantage_vector1.append([value_store[i]])
            value_vector1.append([value_store[i]])

        if play > 150:
            cut = vector_len1 - play + 150
            state_vector1 = state_vector1[:cut]
            action_vector1 = action_vector1[:cut]
            advantage_vector1 = advantage_vector1[:cut]
            value_vector1 = value_vector1[:cut]
            old_mu_vector1 = old_mu_vector1[:cut]
            old_std_vector1 = old_std_vector1[:cut]
            vector_len1 = cut


        print("InitialEpoch : " + str(epoch))

    for epoch in range(1, 50001):
        curreward1 = 0.
        cur_played = 0
        play_step = 0
        first = True
        mu_log = []
        std_log = []
        while cur_played < 1000:
            state = env1.reset()
            reward_store = []
            ex_value_store = []
            for play in range(200):
                prev_state = state
                mu, std = learner1.get_action_stochastic(state)
                if play_step % 2 == 0:
                    action = np.clip(mu + np.exp(std) * np.random.randn(action_dim), -1.0, 1.0)
                else:
                    action = np.random.uniform(-0.5, 0.5, (action_dim))
                mu_log.append(abs(mu))
                std_log.append(std)
                value = learner1.get_expected_value(state)
                state, reward, done, _ = env1.step(action)
                if(done):
                    reward -= 10.
                state_vector1.append(prev_state)
                action_vector1.append(action)
                old_mu_vector1.append(mu)
                old_std_vector1.append(std)
                reward_store.append(reward)
                ex_value_store.append(value)

                curreward1 += reward
                vector_len1 += 1
                cur_played += 1

                if play_step == 0:
                    env1.render()
                if(done):
                    break

            reward_store = np.array(reward_store)
            value_store = np.array(reward_store)
            for backprop in range(1, (30 if play > 31 else play - 1)):
                value_store[:-backprop] += reward_store[backprop:] * (gamma ** backprop)
            for i in range(len(reward_store)):
                advantage_vector1.append([value_store[i] - ex_value_store[i]])
                value_vector1.append([value_store[i]])

            if play > 150:
                cut = vector_len1 - play + 150
                state_vector1 = state_vector1[:cut]
                action_vector1 = action_vector1[:cut]
                advantage_vector1 = advantage_vector1[:cut]
                value_vector1 = value_vector1[:cut]
                old_mu_vector1 = old_mu_vector1[:cut]
                old_std_vector1 = old_std_vector1[:cut]
                vector_len1 = cut
            play_step += 1

        curreward1 /= play_step
        print("Reward1 : " + str(curreward1) + " Step : " + str(play))
        print("Mu : " + str(np.mean(mu_log)) + " Std : " + str(np.mean(std_log)))

        vs1 = 0.
        ps1o = 0.
        ps1n = 0.
        ps1s = 0.

        for history in range(8):
            dic = random.sample(range(vector_len1), 256)
            state_vector_dic = np.array([state_vector1[x] for x in dic])
            value_vector_dic = np.array([value_vector1[x] for x in dic])
            v1 = learner1.optimize_value_batch(state_vector_dic, value_vector_dic)
            vs1 += v1

        history = 0
        while history < 8:

            dic = random.sample(range(vector_len1), 256)
            state_vector_dic = np.array([state_vector1[x] for x in dic])
            action_vector_dic = np.array([action_vector1[x] for x in dic])
            advantage_vector_dic = np.array([advantage_vector1[x] for x in dic])
            old_mu_dic = np.array([old_mu_vector1[x] for x in dic])
            old_std_dic = np.array([old_std_vector1[x] for x in dic])
            advantage_vector_dic = (advantage_vector_dic - np.mean(advantage_vector_dic)) / (np.std(advantage_vector_dic) + 1e-8)

            p1o, p1n, p1s = learner1.optimize_policy_batch(state_vector_dic, action_vector_dic, advantage_vector_dic, old_mu_dic, old_std_dic)
            if p1n is not None:
                ps1o += p1o
                ps1n += p1n
                ps1s += p1s
                history += 1
        
        vs1 /= 8
        ps1o /= 8
        ps1n /= 8
        ps1s /= 8
        print("Epoch " + str(epoch) + " Learner 1 Mean Value Training : " + str(vs1))
        print("Epoch " + str(epoch) + " Learner 1 Mean Policy Training : " + str(ps1o) + " -> " + str(ps1n) + ", " + str(ps1s))


    
        vec_trunc = vector_len1 // 50
        state_vector1 = state_vector1[vec_trunc:]
        action_vector1 = action_vector1[vec_trunc:]
        advantage_vector1 = advantage_vector1[vec_trunc:]
        value_vector1 = value_vector1[vec_trunc:]
        old_mu_vector1 = old_mu_vector1[vec_trunc:]
        old_std_vector1 = old_std_vector1[vec_trunc:]
        vector_len1 -= vec_trunc


        log_file.write(str(epoch) + "\t" +
            str(curreward1) + "\t" + str(np.mean(mu_log)) + "\t" + str(np.mean(std_log)) + "\t" + str(vs1) + "\t" + str(ps1o) + "\t" + str(ps1n) + "\t" + str(ps1s) + "\n")
        if epoch % 1000 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()