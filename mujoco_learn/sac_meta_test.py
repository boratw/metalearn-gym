import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant_1 import AntEnv_1
from gym.envs.mujoco.ant_2 import AntEnv_2
from gym.envs.mujoco.ant_3 import AntEnv_3
import xml.etree.ElementTree as elemTree

from networks.sac_learner2 import SACLearner

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetGoal(goal, name) :
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

    tree.write("../gym/envs/mujoco/assets/" + name)

SetGoal([1.0, 1.0, 1.0, 1.0], "ant_1.xml")
SetGoal([0.01, 1.0, 1.0, 1.0], "ant_2.xml")
SetGoal([1.0, 1.0, 0.01, 1.0], "ant_3.xml")

env1 = AntEnv_1()
env2 = AntEnv_2()
env3 = AntEnv_3()
state_dim = env1.get_current_obs().size
action_dim = env1.action_space.shape[0]
gamma = 0.96

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/meta_sac2_test/"

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] <= 1.0 and state[2] >= 0.27
    return not notdone

learner1 = SACLearner(state_dim, action_dim, name="1", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)
learner2 = SACLearner(state_dim, action_dim, name="2", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)
learner3 = SACLearner(state_dim, action_dim, name="3", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)


with sess.as_default():
    saver.restore(sess, "data/meta_sac2/log_50000.ckpt")
    '''
    for envi, env in enumerate([env1, env2, env3]) :
        for learneri, learner in enumerate([learner1, learner2, learner3]):
            for epoch in range(10):
                log_file = open(LOG_DIR + "log_env" + str(envi) + "_" + str(epoch + 1) + ".txt", "wt")
                log_file.write("Reward\tLearner1_Diff\tLearner2_Diff\tLearner3_Diff\n")
                state = env.reset()
                for play in range(200):

                    action = learner.get_action_stochastic(state)[0]
                    state_est1 = learner1.get_next_state(state, action)
                    state_est2 = learner2.get_next_state(state, action)
                    state_est3 = learner3.get_next_state(state, action)
                    state, reward, done, _ = env.step(action)

                    diff1 = np.mean((state - state_est1) ** 2)
                    diff2 = np.mean((state - state_est2) ** 2)
                    diff3 = np.mean((state - state_est3) ** 2)
                    print(str(reward) + "\t" + str(diff1) + "\t" + str(diff2) + "\t" + str(diff3))
                    log_file.write(str(reward) + "\t" + str(diff1) + "\t" + str(diff2) + "\t" + str(diff3) + "\n")

                    if(done):
                        break
    '''
    log_file = open(LOG_DIR + "log.txt", "wt")
    for envi, env in enumerate([env1, env2, env3]) :
        for learneri, learner in enumerate([learner1, learner2, learner3]):
            curreward = 0
            for epoch in range(20):
                state = env.reset()
                for play in range(200):
                    action = learner.get_action_stochastic(state)[0]
                    state, reward, done, _ = env.step(action)
                    curreward += reward
                    if(done):
                        break
            curreward /= 20
            print("Env " + str(envi) + " Learner " + str(learneri) + "\t" + str(curreward))
            log_file.write("Env " + str(envi) + " Learner " + str(learneri) + "\t" + str(curreward) + "\n")