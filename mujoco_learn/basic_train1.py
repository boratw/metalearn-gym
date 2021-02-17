import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant import AntEnv
from dummyenv import DummyEnv

from networks.sac_learner import SACLearner

env = DummyEnv()
state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/basic_dummy1/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = SACLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    learner.value_network_initialize()

    state_vector = []
    next_state_vector = []
    action_vector = []
    reward_vector = []
    survive_vector = []

    for epoch in range(200):
        state = env.reset()
        for play in range(200):
            prev_state = state

            state_vector.append(state)
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env.step(action)
            action_vector.append(action)
            reward_vector.append([reward])
            survive_vector.append([ 0.0 if done else 1.0])
            next_state_vector.append(state)
            if(done):
                break
                

        print("InitialEpoch : " + str(epoch))

    for epoch in range(1, 20001):
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
        print("Epoch Discrete : " + str(epoch) + "\tStep: " + str(play) + "\tReward: " + str(rewards))
        
        for _ in range(8):
            rewards = 0.
            state = env.reset()
            for play in range(200):
                prev_state = state

                state_vector.append(state)
                action = learner.get_action_stochastic(state)[0]
                state, reward, done, _ = env.step(action)
                
                action_vector.append(action)
                reward_vector.append([reward])
                survive_vector.append([ 0.0 if done else 1.0])
                next_state_vector.append(state)

                rewards += reward

                if(done):
                    break
            reward_stoc += rewards
            print("Epoch Stocastic : " + str(epoch) + "\tStep: " + str(play) + "\tReward: " + str(rewards))
        '''
        state = env.reset()
        for play in range(200):
            prev_state = state

            state_vector.append(state)
            action = learner.get_action(state)[0]
            state, reward, done, _ = env.step(action)
            
            action_vector.append(action)
            reward_vector.append([reward])
            next_state_vector.append(state)
            env.render()


            print("Epoch : " + str(epoch) + "\tStep: " + str(play) + "\tReward: " + str(reward))
            if(done):
                break
        '''
        print("=============================================")
        print("Reward_Discrete : " + str(reward_disc))
        print("Reward_Stocastic : " + str(reward_stoc / 8))

        veclen = len(state_vector)
        qs = []
        vs = []
        ps = []
        for history in range(64):
            dic = random.sample(range(veclen), 32)

            state_vector_dic = [state_vector[x] for x in dic]
            next_state_vector_dic = [next_state_vector[x] for x in dic]
            action_vector_dic = [action_vector[x] for x in dic]
            reward_vector_dic = [reward_vector[x] for x in dic]
            survive_vector_dic = [survive_vector[x] for x in dic]

            q, v, p = learner.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs.extend(q)
            vs.extend(v)
            ps.extend(p)
        learner.value_network_update()


        vec_trunc = veclen // 20
        state_vector = state_vector[vec_trunc:]
        next_state_vector = next_state_vector[vec_trunc:]
        action_vector = action_vector[vec_trunc:]
        reward_vector = reward_vector[vec_trunc:]
        survive_vector = survive_vector[vec_trunc:]


        print("Epoch : " + str(epoch) 
            + "\t" + str(np.mean(qs)) + "\t" + str(np.std(qs))
            + "\t" + str(np.mean(vs)) + "\t" + str(np.std(vs))
            + "\t" + str(np.mean(ps)) + "\t" + str(np.std(ps)))
        print("=============================================")


        log_file.write("Episode\t" + str(epoch) + "\tScore\t" + str(reward_disc) +
            "\tQvalue\t" + str(np.mean(qs)) + "\t" + str(np.std(qs)) +
            "\tValue\t" + str(np.mean(vs)) + "\t" + str(np.std(vs)) +
            "\tPolicy_Pi\t" + str(np.mean(ps)) + "\t" + str(np.std(ps)) +"\n")
        if epoch % 100 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()