import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant6_modified import Ant6ModifiedEnv
from gym.envs.mujoco.ant6 import Ant6Env
from networks.trpo2_learner import TRPOLearner
from networks.maml2_learner import MAMLLearner
from dummyenv import DummyEnv

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetGoal(goal) :
    tree = elemTree.parse("../gym/envs/mujoco/assets/ant6.xml")
    for body in tree.iter("body"):
        if "name" in body.attrib:
            if(body.attrib["name"] == "aux_1"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[0])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
            if(body.attrib["name"] == "aux_2"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[1])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
            if(body.attrib["name"] == "aux_3"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[2])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
            if(body.attrib["name"] == "aux_4"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[3])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])
            if(body.attrib["name"] == "aux_5"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[4])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[4])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[4])
            if(body.attrib["name"] == "aux_6"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[5])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[5])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[5])

    tree.write("../gym/envs/mujoco/assets/ant6_modified.xml")


env = Ant6Env()

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/ol3/"
log_file = open(LOG_DIR + "log.txt", "wt")

mamllearner = TRPOLearner(state_dim, action_dim)
trpolearners = [TRPOLearner(state_dim, action_dim, name="0")]

sess = tf.Session()
#saver = tf.train.Saver(max_to_keep=0)


tasks = [   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.01, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 0.01, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.01, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   ]

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch_count = 0
    mean_reward = 0.0

    current_agent = 0

    state_maml_vector = []
    action_maml_vector = []
    reward_maml_vector = []
    nextstate_maml_vector = []

    for task in tasks:
        SetGoal(task)
        env.close()
        env = Ant6ModifiedEnv()

        for epoch in range(20):
            epoch_count += 1
            log_rewards = 0.
            log_steps = 0.
            log_learning_rate = 0.
            log_loss_policy = 0.
            log_loss_value = 0.
            log_divergences = 0.


            state_vector = []
            action_vector = []
            reward_vector = []
            nextstate_vector = []
        
            sigma_rate = 0.0
            prevreward = 0.
            state_corr = [0.] * (len(trpolearners) + 1)

            for batch in range(8):
                state = env.reset()
                curreward = 0.
                reward = 0.
                batchstart = len(state_vector)
                for play in range(201):

                    prevstate = state
                    action = trpolearners[current_agent].get_action_stochastic(state, sigma)[0]
                    state, reward, done, d = env.step(action)

                    state_vector.append(prevstate)
                    action_vector.append(action)
                    reward_vector.append([reward])
                    nextstate_vector.append(state)

                    if play == 0:
                        state_maml_vector.append(prevstate)
                        action_maml_vector.append(action)
                        reward_maml_vector.append([reward])
                        nextstate_maml_vector.append(state)
                        


                    for i in range(len(trpolearners))
                        state_corr[i] += trpolearners[i].get_next_diff(prevstate, action, state)[0]
                    state_corr[len(trpolearners)] += mamllearner.get_next_diff(prevstate, action, state)[0]

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

                print("Epoch " + str(epoch) + " SigmaRate:" + str(sigma_rate) + " Reward: " + str(curreward) + " Step:" + str(play))
                print("Corr : " + str(state_corr))

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

            log_rewards /= 8
            log_steps /= 8

            overfitted = False
            learning_rate = 0.01
            while True:

                l, d = trpolearners[current_agent].optimize_policy_batch(learning_rate, state_vector, action_vector, value_vector, nextstate_vector)
                
                print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
                if d > 1. or np.isnan(d):
                    overfitted = True
                    learning_rate /= 2.
                else:
                    if overfitted:
                        break
                    else:
                        learning_rate *= 2.
            trpolearners[current_agent].optimize_end()

            state_corr = [np.exp(-corr) for corr in state_corr]
            state_corr /= np.sum(state_corr)
            print("Prob : " + str(state_corr))

            for i in range(len(trpolearners))
                learner.optimize_value_batch(state_corr[i] * 0.01, state_vector, nextstate_vector, value_vector)
                learner.optimize_nextstate_batch(state_corr[i] * 0.01, state_vector, action_vector, nextstate_vector)

            log_loss_value /= 32
            log_loss_nextstate /= 32
            print("Value loss " + str(log_loss_value))
            print("Nextstate loss " + str(log_loss_nextstate))


            if current_agent == -1:
                trpolearners.append(TRPOLearner(state_dim, action_dim, name=str(len(trpolearners))))
                trpolearners[-1].init_from_maml(mamllearner)
                trpoprobs.append(0.5)
                current_agent = len(trpolearners) - 1
                print("New Agent")

env.close()