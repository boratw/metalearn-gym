import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant_modified import AntModifiedEnv
from gym.envs.mujoco.ant import AntEnv
from networks.maml_learner import MAMLLearner
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
            if(body.attrib["name"] == "aux_3"):
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

LOG_DIR = "data/maml1/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = MAMLLearner(state_dim, action_dim, value_gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(1, 20001):
        log_sum_rewards = 0.
        log_sum_steps = 0.
        log_sum_mu = 0.
        log_sum_sigma = 0.
        log_sum_loss_policy = 0.
        log_sum_loss_value = 0.
        log_sum_loss_nextstate = 0.
        log_sum_divergences = 0.

        learner.gradient_init()
        for envi in range(4):
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

            env.close()
            g = [np.random.uniform(0.75, 1.25), np.random.uniform(0.75, 1.25)]
            SetGoal(g)
            print("Epoch " + str(epoch) + " Env " + str(envi) + " Goal: " + str(g))
            env = AntModifiedEnv()

            sigma_rate = 0.0
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
                    if(done):
                        break
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

                log_mu += mu / play
                log_sigma += sigma / play
                log_rewards += curreward
                log_steps += play
                print("Epoch " + str(epoch) + " Env " + str(envi) + " SigmaRate:" + str(sigma_rate) + " Reward: " + str(curreward) + " Step:" + str(play))

                if batch != 0:
                    if prevreward < curreward or prevreward < 50. :
                        sigma_rate += 0.05
                        prevreward = curreward
                else:
                    log_rewards += curreward
                    log_steps += play
                    if curreward < 25.:
                        sigma_rate += 0.05

                    prevreward = curreward

            log_rewards /= 8
            log_steps /= 8
            log_mu /= 8.
            log_sigma /= 8.
            print("Log_Mu: " + str(log_mu) + " Log_Sigma: " + str(log_sigma))

            learner.optimize_start()

            for i in range(32):
                dic = random.sample(range(len(state_vector)), 64 if len(state_vector) >= 64 else len(state_vector))
                state_vector_dic = [state_vector[x] for x in dic]
                action_vector_dic = [action_vector[x] for x in dic]
                value_vector_dic = [value_vector[x] for x in dic]
                nextstate_vector_dic = [nextstate_vector[x] for x in dic]

                v = learner.optimize_value_batch(state_vector_dic, nextstate_vector_dic, value_vector_dic)
                log_loss_value += v
                n = learner.optimize_nextstate_batch(state_vector_dic, action_vector_dic, nextstate_vector_dic)
                log_loss_nextstate += n
            log_loss_value /= 32
            log_loss_nextstate /= 32
            print("Value loss " + str(log_loss_value))
            print("Nextstate loss " + str(log_loss_nextstate))

            overfitted = False
            learning_rate = 0.01
            while True:
                dic = random.sample(range(len(state_vector)), 256 if len(state_vector) >= 256 else len(state_vector))
                state_vector_dic = [state_vector[x] for x in dic]
                action_vector_dic = [action_vector[x] for x in dic]
                value_vector_dic = [value_vector[x] for x in dic]
                nextstate_vector_dic = [nextstate_vector[x] for x in dic]

                l, d, diff = learner.optimize_policy_batch(learning_rate, state_vector_dic, action_vector_dic, value_vector_dic, nextstate_vector_dic)
                
                print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d), " Loss Update:" + str(diff) )
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
            log_divergences = d

            print("Policy Loss " + str(log_loss_policy))
            print("Divergence " + str(log_divergences))


            log_sum_rewards += log_rewards
            log_sum_steps += log_steps
            log_sum_mu += log_mu
            log_sum_sigma += log_sigma
            log_sum_loss_policy += log_loss_policy
            log_sum_loss_value += log_loss_value
            log_sum_loss_nextstate += log_loss_nextstate
            log_sum_divergences += log_divergences

        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_sum_rewards / 4) +
            "\tSteps\t" + str(log_sum_steps / 4) +
            "\tOutput_Mu\t" + str(log_sum_mu / 4) +
            "\tOutput_Sigma\t" + str(log_sum_sigma / 4) +
            "\tLossValue\t" + str(log_sum_loss_value / 4) +
            "\tLossNextState\t" + str(log_sum_loss_nextstate / 4) +
            "\tLossPolicy\t" + str(log_sum_loss_policy / 4) +
            "\tDivergence\t" + str(log_sum_divergences / 4) + "\n")


        learner.gradient_assign()



        if epoch % 100 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()