import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant_modified import AntModifiedEnv
from gym.envs.mujoco.ant6 import Ant6Env
from networks.trpo_learner import TRPOLearner
from dummyenv import DummyEnv

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetGoal(goal) :
    tree = elemTree.parse("../gym/envs/mujoco/assets/ant.xml")
    for body in tree.iter("body"):
        if "name" in body.attrib:
            if(body.attrib["name"] == "aux_1"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[0])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
            if(body.attrib["name"] == "aux_4"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[1])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])

    tree.write("../gym/envs/mujoco/assets/ant_modified.xml")


env = Ant6Env()

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/trpo4/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = TRPOLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)


    mean_reward = 0.0

    for epoch in range(1, 5001):
        log_rewards = 0.
        log_steps = 0.
        log_learning_rate = 0.
        log_loss_policy = 0.
        log_loss_value = 0.
        log_divergences = 0.

        #env.close()
        #SetGoal(np.random.uniform(0.5, 1.5, 2))
        #env = AntModifiedEnv()
        #env.set_goal(2.0)
        
        #learner.init_policies()
        state_vector = []
        action_vector = []
        reward_vector = []
        sigma = 0.069291665
        for batch in range(4):
            batchstart = len(state_vector)
            state = env.reset()
            meaned_reward = mean_reward / 200.
            curreward = 0.
            reward = 0.
            #rand = False
            #randomstart = 0.001 * 1.5 ** batch
            for play in range(201):
                #if batch != 0:
                #   if rand:
                #        if random.random() < 0.05:
                #            rand = False
                #    else:
                #        if random.random() < randomstart:
                #            rand = True
                prevstate = state
                #if rand:
                #    action = np.random.uniform(-1.0, 1.0, action_dim)
                #else:
                action = learner.get_action_stochastic(state, sigma)[0]
                state, reward, done, d = env.step(action)

                state_vector.append(prevstate)
                action_vector.append(action)
                reward_vector.append([reward - meaned_reward])

                curreward += reward
                if batch == 0:
                    env.render()
                if(done):
                    break
            sigma *= 1.25

            #if play == 200:
            #    reward_vector[-1][0] +=  (curreward / 200) * (1. / (1. / 0.96) - 1.)
            #else:
            #reward_quality = np.exp(np.clip((curreward - mean_reward) / mean_reward, -2., 2.))
            if play != 200:
                reward_vector[-1][0] -= 10.0
            for i in range(len(reward_vector) - 1, batchstart, -1):
                reward_vector[i-1][0] =  0.96 * reward_vector[i][0] + reward_vector[i-1][0]
            #for i in range(batchstart, len(reward_vector)):
            #    reward_vector[i][0] = reward_vector[i][0] * reward_quality

            log_rewards += curreward
            log_steps += play
            print("Epoch " + str(epoch) + " Reward: " + str(curreward) + " Step:" + str(play))

        log_rewards /= 4.
        log_steps /= 4.

        mean_reward = mean_reward * 0.9 + log_rewards * 0.1

        #learner.change_policy(state_vector, action_vector, reward_vector)

        learning_rate = 1.
        overfitted = False
        while True:

            dic = random.sample(range(len(state_vector)), 512 if len(state_vector) >= 512 else len(state_vector))
            state_vector_dic = [state_vector[x] for x in dic]
            action_vector_dic = [action_vector[x] for x in dic]
            reward_vector_dic = [reward_vector[x] for x in dic]
            l, d = learner.optimize_policy_batch(learning_rate, state_vector_dic, action_vector_dic, reward_vector_dic)
            
            print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
            if d > 0.01:
                overfitted = True
                learning_rate /= 2.
            else:
                if overfitted:
                    break
                else:
                    learning_rate *= 2.

        learner.update_policy()
        log_loss_policy += l
        log_divergences += d
        log_learning_rate += learning_rate
            
        print("mean_reward " + str(mean_reward))

        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_rewards) +
            "\tSteps\t" + str(log_steps) +
            "\tLearningRate\t" + str(log_learning_rate) +
            "\tLoss\t" + str(log_loss_policy) +
            "\tDivergence\t" + str(log_divergences ) + "\n")


        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()