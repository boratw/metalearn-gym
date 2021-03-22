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

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/maml1/"
log_file = open(LOG_DIR + "log.txt", "wt")

learner = MAMLLearner(state_dim, action_dim)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    mean_reward = 0.

    state_vector = []
    action_vector = []
    reward_vector = []


    for epoch in range(1, 3001):
        log_rewards = 0.
        log_learning_rate = 0.
        log_losses = 0.
        log_divergences = 0.
        log_steps = 0.

        validate_result = np.zeros((4))
        #learner.initialize_policy()

        for envi in range(4):

            env.close()
            SetGoal(np.random.uniform(0.75, 1.25, 2))
            env = AntModifiedEnv()
            #env.set_goal(2.0)
            

            rewards = 0.
            losses = 0.
            divergences = 0.
            steps = 0.
            sigma = 0.069291665

            for batch in range(4):
                batchstart = len(state_vector)
                state = env.reset()
                curreward = 0.
                reward = 0.
                for play in range(201):
                    state_vector.append(state)

                    action = learner.get_action_stochastic_initial(state, sigma=sigma)[0]
                    state, reward, done, _ = env.step(action)

                    action_vector.append(action)
                    reward_vector.append([reward])
                    curreward += reward

                    if(done):
                        break
                    #if batch == 0:
                    #    env.render()

                if play == 200:
                    reward_vector[-1][0] +=  (curreward / 200) * (1. / (1. / 0.95) - 1.)
                else:
                    reward_vector[-1][0] -= 1.0

                for i in range(len(reward_vector) - 1, batchstart + 1, -1):
                    reward_vector[i-1][0] =  0.95 * reward_vector[i][0] + reward_vector[i-1][0]

                rewards += curreward
                steps += play
                print("Epoch " + str(epoch) + " Env " + str(envi) + " Reward: " + str(curreward) + " Step:" + str(play))

            rewards /= 4.
            steps /= 4.
            log_steps += steps
            log_rewards += rewards

            for update in range(4):
                learning_rate = 1.
                overfitted = False
                while True:
                    dic = random.sample(range(len(state_vector)), 256 if len(state_vector) >= 256 else len(state_vector))
                    state_vector_dic = [state_vector[x] for x in dic]
                    action_vector_dic = [action_vector[x] for x in dic]
                    reward_vector_dic = [reward_vector[x] for x in dic]
                    l, d = learner.optimize_policy_batch(envi, learning_rate, state_vector_dic, action_vector_dic, reward_vector_dic)
                    print("Epoch " + str(epoch) + " Env " + str(envi) + " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )

                    if d > 0.01:
                        overfitted = True
                        learning_rate /= 2.
                    else:
                        if overfitted:
                            break
                        else:
                            learning_rate *= 2.

                log_losses += l
                log_divergences += d
                log_learning_rate += learning_rate


            mean_reward = mean_reward * 0.95 + rewards * 0.05
            print("mean_reward " + str(mean_reward))
            '''
            print("Epoch " + str(epoch) + " Env " + str(envi) + " Update " + str(update) + " Loss: " + str(l) + " Divergence:" + str(d))
            log_losses += l
            log_divergences += d

            curreward = 0.

            state = env.reset()
            for batch in range(8):
                for play in range(200):
                    action = learner.get_action_deterministic(envi, state)[0]
                    state, reward, done, _ = env.step(action)

                    curreward += reward
                    if(done):
                        break
            initreward = 0.

            state = env.reset()
            for batch in range(8):
                for play in range(200):
                    action = learner.get_action_deterministic_initial(state)[0]
                    state, reward, done, _ = env.step(action)

                    initreward += reward
                    if(done):
                        break
            curreward = (curreward - initreward) / 8.
            log_reward_advance += curreward

            print("Epoch " + str(epoch) + " Env " + str(envi) + " Validation " + str(curreward))
            if curreward < 0.1:
                curreward = 0.1

            validate_result[envi] = np.log(curreward + 1.)
            '''
        log_file.write("Epoch\t" + str(epoch) + 
            "\tReward\t" + str(log_rewards / 16) +
            "\tSteps\t" + str(log_steps / 16) +
            "\tLearningRate\t" + str(log_learning_rate / 16) +
            "\tLoss\t" + str(log_losses / 16) +
            "\tDivergence\t" + str(log_divergences / 16) + "\n")

        learner.optimize_meta_policy()



        if epoch % 50 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")

        length = len(state_vector) // 20
        state_vector = state_vector[length:]
        action_vector = action_vector[length:]
        reward_vector = reward_vector[length:]

env.close()