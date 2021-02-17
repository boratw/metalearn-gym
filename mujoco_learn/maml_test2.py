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


learner = MAMLLearner(state_dim, action_dim, value_gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    saver.restore(sess, "./data/maml1/log_1000.ckpt")

    for epoch in range(1, 101):
        env.close()
        g = np.random.uniform(0.5, 1.5, 2)
        SetGoal(g)
        env = AntModifiedEnv()
        for batch in range(16):
            state = env.reset()
            for play in range(201):
                action = learner.get_action_stochastic(state, 1.5)[0]
                state, reward, done, _ = env.step(action)
                env.render()
                if(done):
                    break
env.close()