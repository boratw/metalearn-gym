import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant import AntEnv
import xml.etree.ElementTree as elemTree

from networks.sac_learner import SACLearner

env1 = AntEnv()
state_dim = env1.get_current_obs().size
action_dim = env1.action_space.shape[0]
gamma = 0.96

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/basic_train5/"
log_file = open(LOG_DIR + "log.txt", "wt")

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] <= 1.0 and state[2] >= 0.27
    return not notdone

learner1 = SACLearner(state_dim, action_dim, name="1", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

log_file.write("Episode\t" +
    "Env1_Score\tEnv1_Qvalue\tEnv1_Value\tEnv1_Policy\n")

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    learner1.value_network_initialize()

    vector_len1 = 0
    state_vector1 = []
    next_state_vector1 = []
    action_vector1 = []
    reward_vector1 = []
    survive_vector1 = []
    value_vector1 = []

    for epoch in range(100):
        state = env1.reset()
        reward_store = []
        for play in range(200):
            prev_state = state
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env1.step(action)
            state_vector1.append(prev_state)
            action_vector1.append(action)
            survive_vector1.append([ 0.0 if done else 1.0])
            next_state_vector1.append(state)
            reward_store.append(reward)

            vector_len1 += 1
            if(done):
                break

        value_store = np.array(reward_store)
        for backprop in range(1, (20 if play > 21 else play - 1)):
            value_store[:-backprop] += value_store[backprop:] * (gamma ** backprop)
        for i in range(len(reward_store)):
            reward_vector1.append([reward_store[i]])
            value_vector1.append([value_store[i]])

        print("InitialEpoch : " + str(epoch))

    for epoch in range(1, 50001):


        curreward1 = 0.
        state = env1.reset()
        reward_store = []
        for play in range(200):
            prev_state = state

            action = learner1.get_action_stochastic(state)[0]
            state, reward, done, _ = env1.step(action)
            
            state_vector1.append(prev_state)
            action_vector1.append(action)
            survive_vector1.append([ 0.0 if done else 1.0])
            next_state_vector1.append(state)
            reward_store.append(reward)

            vector_len1 += 1
            if(done):
                break
            curreward1 += reward

            env1.render()

        value_store = np.array(reward_store)
        for backprop in range(1, (20 if play > 21 else play - 1)):
            value_store[:-backprop] += value_store[backprop:] * (gamma ** backprop)
        for i in range(len(reward_store)):
            reward_vector1.append([reward_store[i]])
            value_vector1.append([value_store[i]])
        print("Reward1 : " + str(curreward1) + " Step : " + str(play))

        qs1 = 0.
        vs1 = 0.
        ps1 = 0.
        for history in range(8):
            dic = random.sample(range(vector_len1), 256)

            state_vector_dic = [state_vector1[x] for x in dic]
            next_state_vector_dic = [next_state_vector1[x] for x in dic]
            action_vector_dic = [action_vector1[x] for x in dic]
            reward_vector_dic = [reward_vector1[x] for x in dic]
            survive_vector_dic = [survive_vector1[x] for x in dic]

            q, v, p = learner1.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs1 += np.mean(q)
            vs1 += np.mean(v)
            ps1 += np.mean(p)
        learner1.value_network_update()
        qs1 /= 8.
        vs1 /= 8.
        ps1 /= 8.
        print("Epoch " + str(epoch) + " Learner 1 Mean Qvalue Training : " + str(qs1))
        print("Epoch " + str(epoch) + " Learner 1 Mean Value Training : " + str(vs1))
        print("Epoch " + str(epoch) + " Learner 1 Mean Policy Training : " + str(ps1))


        vec_trunc = vector_len1 // 20
        state_vector1 = state_vector1[vec_trunc:]
        next_state_vector1 = next_state_vector1[vec_trunc:]
        action_vector1 = action_vector1[vec_trunc:]
        reward_vector1 = reward_vector1[vec_trunc:]
        survive_vector1 = survive_vector1[vec_trunc:]
        value_vector1 = value_vector1[vec_trunc:]
        vector_len1 -= vec_trunc


        log_file.write(str(epoch) + "\t" +
            str(curreward1) + "\t" + str(qs1) + "\t" + str(vs1) + "\t" + str(ps1) + "\n")
        if epoch % 1000 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()