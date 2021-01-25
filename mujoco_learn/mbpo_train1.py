import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant import AntEnv

from networks.sac_learner import SACLearner
from networks.model_env import ModelEnv

def survive(state):
    return state[2] >= 0.3 and state[2] <= 1.0


env = AntEnv()

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]
model_env = ModelEnv(state_dim, action_dim, survive)


print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/mbpo1/"
log_file = open(LOG_DIR + "log1.txt", "wt")

learner = SACLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    learner.value_network_initialize()

    state_true_vector = []
    next_state_true_vector = []
    action_true_vector = []
    reward_true_vector = []
    survive_true_vector = []

    state_model_vector = []
    next_state_model_vector = []
    action_model_vector = []
    reward_model_vector = []
    survive_model_vector = []

    for epoch in range(200):
        state = env.reset()
        for play in range(200):
            prev_state = state

            state_true_vector.append(state)
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env.step(action)
            action_true_vector.append(action)
            reward_true_vector.append([reward])
            survive_true_vector.append([ 0.0 if done else 1.0])
            next_state_true_vector.append(state)
            if(done):
                break
                

        print("InitialEpoch : " + str(epoch))

    for epoch in range(1, 10001):
        reward_disc = 0
        reward_stoc = 0


        state = env.reset()
        
        rewards = 0.
        for play in range(200):
            action = learner.get_action_deterministic(state)[0]
            state, reward, done, _ = env.step(action)
            
            rewards += reward

            #env.render()

            if(done):
                break
        reward_disc = rewards
        
        for _ in range(8):
            rewards = 0.
            state = env.reset()
            for play in range(200):
                prev_state = state

                state_true_vector.append(state)
                action = learner.get_action_stochastic(state)[0]
                state, reward, done, _ = env.step(action)
                
                action_true_vector.append(action)
                reward_true_vector.append([reward])
                survive_true_vector.append([ 0.0 if done else 1.0])
                next_state_true_vector.append(state)

                rewards += reward

                if(done):
                    break
            reward_stoc += rewards
            print("Epoch Stocastic : " + str(epoch) + "\tStep: " + str(play) + "\tReward: " + str(rewards))

        print("=============================================")
        print("Reward_Discrete : " + str(reward_disc))
        print("Reward_Stocastic : " + str(reward_stoc / 8))

        veclen = len(state_true_vector)
        ls = 0.
        for history in range(64):
            dic = random.sample(range(veclen), 32)

            state_vector_dic = [state_true_vector[x] for x in dic]
            next_state_vector_dic = [next_state_true_vector[x] for x in dic]
            action_vector_dic = [action_true_vector[x] for x in dic]
            reward_vector_dic = [reward_true_vector[x] for x in dic]

            l = model_env.optimize(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic)
            ls += l

        dic = random.sample(range(veclen), 256)
        for rollout in range(256):
            state = np.tile(state_true_vector[dic[rollout]], (16, 1))
            action = learner.get_action_stochastic_batch(state)

            next_state, reward, survive  = model_env.get_batch(state, action)

            state_model_vector.extend(state)
            next_state_model_vector.extend(next_state)
            action_model_vector.extend(action)
            reward_model_vector.extend(reward)
            survive_model_vector.extend(survive)



        veclen = len(state_model_vector)
        qs = 0.
        vs = 0.
        ps = 0.
        for history in range(64):
            dic = random.sample(range(veclen), 32)

            state_vector_dic = [state_model_vector[x] for x in dic]
            next_state_vector_dic = [next_state_model_vector[x] for x in dic]
            action_vector_dic = [action_model_vector[x] for x in dic]
            reward_vector_dic = [reward_model_vector[x] for x in dic]
            survive_vector_dic = [survive_model_vector[x] for x in dic]

            q, v, p = learner.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs += np.mean(q)
            vs += np.mean(v)
            ps += np.mean(p)
        learner.value_network_update()


        vec_trunc = veclen // 20
        state_true_vector = state_true_vector[vec_trunc:]
        next_state_true_vector = next_state_true_vector[vec_trunc:]
        action_true_vector = action_true_vector[vec_trunc:]
        reward_true_vector = reward_true_vector[vec_trunc:]
        survive_true_vector = survive_true_vector[vec_trunc:]

        state_model_vector = state_model_vector[vec_trunc:]
        next_state_model_vector = next_state_model_vector[vec_trunc:]
        action_model_vector = action_model_vector[vec_trunc:]
        reward_model_vector = reward_model_vector[vec_trunc:]
        survive_model_vector = survive_model_vector[vec_trunc:]

        print("Epoch : " + str(epoch) 
            + "\t" + str(ls / 64)
            + "\t" + str(qs / 64)
            + "\t" + str(vs / 64) 
            + "\t" + str(ps / 64))
        print("=============================================")


        log_file.write("Episode\t" + str(epoch) + "\tScore\t" + str(reward_disc) +
            "\tModelLoss\t" + str(ls / 64) +
            "\tQvalue\t" + str(qs / 64) +
            "\tValue\t" + str(vs / 64) +
            "\tPolicy_Pi\t" + str(ps / 64) +"\n")
        if epoch % 100 == 0:
            saver.save(sess, LOG_DIR + "log1_" + str(epoch) + ".ckpt")


env.close()