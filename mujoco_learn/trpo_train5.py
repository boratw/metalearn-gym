import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant_modified import AntModifiedEnv
from gym.envs.mujoco.ant import AntEnv
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
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[0])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])

    tree.write("../gym/envs/mujoco/assets/ant_modified.xml")

env = AntEnv()
gamma = 0.95

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/trpo5/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = TRPOLearner(state_dim, action_dim, value_gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver.restore(sess, "./data/trpo5/log_150.ckpt")


    mean_reward = 0.0

    for epoch in range(1, 5001):
        log_rewards = 0.
        log_steps = 0.
        log_mu = 0.
        log_sigma = 0.
        log_loss_policy = 0.
        log_loss_value = 0.
        log_loss_nextstate = 0.
        log_divergences = 0.

        state_vector = []
        action_vector = []
        value_vector = []
        nextstate_vector = []

        sigma_rate = 0.0
        prevreward = 0.


        env.close()
        g = [np.random.uniform(0.75, 1.25), np.random.uniform(0.75, 1.25)]
        SetGoal(g)
        print("Epoch " + str(epoch) + " Goal: " + str(g))
        env = AntModifiedEnv()

        for batch in range(8):
            
            batchstart = len(state_vector)
            state = env.reset()
            curreward = 0.
            reward = 0.
            mu = 0.
            sigma = 0.
            for play in range(201):
                prevstate = state

                action, m, s = learner.get_action_stochastic(state, sigma_rate)
                action = action[0]
                mu += np.mean(np.abs(m[0]))
                sigma += np.mean(s[0])
                state, reward, done, _ = env.step(action)

                state_vector.append(prevstate)
                action_vector.append(action)
                nextstate_vector.append(state)
                value_vector.append([reward])

                curreward += reward
                #if batch == 0:
                #    env.render()
                if(done):
                    break
            log_mu += mu / play
            log_sigma += sigma / play
            #if play == 200:
            #    value_vector[-1][0] +=  (np.mean(reward_vector[-10]) / 10.) * (1. / (1. - gamma) - 1.)
            #else:
            if play != 200:
                value_vector[-1][0] -= 1.0

            for i in range(len(value_vector) - 1, batchstart, -1):
                value_vector[i-1][0] =  gamma * value_vector[i][0] + value_vector[i-1][0]

            if play > 150:
                cut = play - 150
                state_vector = state_vector[:-cut]
                action_vector = action_vector[:-cut]
                nextstate_vector = nextstate_vector[:-cut]
                value_vector = value_vector[:-cut]

            print("Epoch " + str(epoch) + " SigmaRate:" + str(sigma_rate) + " Reward: " + str(curreward) + " Step:" + str(play))

            if batch != 0:
                if prevreward < curreward:
                    sigma_rate *= 2
                    prevreward = curreward
            else:
                log_rewards += curreward
                log_steps += play
                if curreward < 25.:
                    sigma_rate += 0.1

                prevreward = curreward

        #log_rewards /= 8.
        #log_steps /= 8.
        log_mu /= 8.
        log_sigma /= 8.
        print("Log_Mu: " + str(log_mu) + " Log_Sigma: " + str(log_sigma))

        overfitted = False
        learning_rate = 0.01
        while True:
            dic = random.sample(range(len(state_vector)), 256 if len(state_vector) >= 256 else len(state_vector))
            state_vector_dic = [state_vector[x] for x in dic]
            action_vector_dic = [action_vector[x] for x in dic]
            value_vector_dic = [value_vector[x] for x in dic]
            nextstate_vector_dic = [nextstate_vector[x] for x in dic]

            l, d = learner.optimize_policy_batch(learning_rate, state_vector_dic, action_vector_dic, value_vector_dic, nextstate_vector_dic)
            
            print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
            if d > 100. or np.isnan(d):
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
            action_vector_dic = [action_vector[x] for x in dic]
            nextstate_vector_dic = [nextstate_vector[x] for x in dic]

            v = learner.optimize_value_batch(state_vector_dic, nextstate_vector_dic, value_vector_dic)
            log_loss_value += v
            n = learner.optimize_nextstate_batch(state_vector_dic, action_vector_dic, nextstate_vector_dic)
            log_loss_nextstate += n
        log_loss_value /= 32
        log_loss_nextstate /= 32
        print("Value loss " + str(log_loss_value))
        print("Nextstate loss " + str(log_loss_nextstate))

        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_rewards) +
            "\tSteps\t" + str(log_steps) +
            "\tOutput_Mu\t" + str(log_mu) +
            "\tOutput_Sigma\t" + str(log_sigma) +
            "\tLossValue\t" + str(log_loss_value) +
            "\tLossNextstate\t" + str(log_loss_nextstate) +
            "\tLossPolicy\t" + str(log_loss_policy) +
            "\tDivergence\t" + str(log_divergences ) + "\n")


        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()