import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant6_modified import Ant6ModifiedEnv
from gym.envs.mujoco.ant6 import Ant6Env
from networks.trpo_learner import TRPOLearner
from networks.maml_learner import MAMLLearner
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

LOG_DIR = "data/ol1/"
log_file = open(LOG_DIR + "log.txt", "wt")

mamllearner = MAMLLearner(state_dim, action_dim)
trpolearners = [TRPOLearner(state_dim, action_dim, name="0")]
trpoprobs = [1.5]

sess = tf.Session()
#saver = tf.train.Saver(max_to_keep=0)


tasks = [   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.01, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 0.01, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.01, 1.0, 1.0, 1.0, 1.0],
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

        for epoch in range(100):
            epoch_count += 1
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
            nextstate_vector = []
        
            meaned_reward = mean_reward / 200.
            sigma = 0.069291665
            for batch in range(4):
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
                    reward_vector.append([reward - meaned_reward])
                    nextstate_vector.append(state)

                    curreward += reward
                    if(done):
                        break
                    
                    #if batch == 0:
                    #    env.render()

                if play != 200:
                    reward_vector[-1][0] -= 10.0
                for i in range(len(reward_vector) - 1,  batchstart, -1):
                    reward_vector[i-1][0] =  0.96 * reward_vector[i][0] + reward_vector[i-1][0]

                log_rewards += curreward
                log_steps += play
                print("Epoch " + str(epoch_count) +" Agent: " + str(current_agent) + " Reward: " + str(curreward) + " Step:" + str(play))
                sigma *= 1.25

            log_rewards /= 4
            log_steps /= 4

            state_maml_vector.append(state_vector)
            action_maml_vector.append(action_vector)
            reward_maml_vector.append(reward_vector)
            nextstate_maml_vector.append(nextstate_vector)
            
            if len(state_maml_vector) > 5:
                state_maml_vector = state_maml_vector[1:]
                action_maml_vector = action_maml_vector[1:]
                reward_maml_vector = reward_maml_vector[1:]
                nextstate_maml_vector = nextstate_maml_vector[1:]

            mean_reward = mean_reward * 0.9 + log_rewards * 0.1
            print("mean_reward", mean_reward)

            log_file.write("Epoch\t" + str(epoch_count) + 
                "\tReward\t" + str(log_rewards) +
                "\tSteps\t" + str(log_steps) +
                "\tAgent\t" + str(current_agent))
            trpo_scores = [ trpolearner.get_score(nextstate_vector, action_vector, nextstate_vector) for trpolearner in trpolearners]
            mamllearner.init_policy()
            for i in range(len(state_maml_vector) - 1):
                mamllearner.optimize_policy_batch(state_maml_vector[i], action_maml_vector[i], reward_maml_vector[i], nextstate_maml_vector[i])
            mamllearner.update_policy()

            maml_score = mamllearner.get_score(state_vector, action_vector, nextstate_vector)

            score_sum = maml_score
            print("score maml\t" + str(maml_score))
            log_file.write("\tMaml_Score\t" + str(maml_score))
            score_sum = 0.
            prob_sum = 0.
            for i in range(len(trpolearners)):
                print("score trpo" + str(i) + "\t" + str(trpo_scores[i]))
                log_file.write("\tTrpo"+str(i)+"_Score\t" + str(trpo_scores[i]) + "\tTrpo"+str(i)+"_Prob\t" + str(trpoprobs[i]))
                score_sum += trpo_scores[i]
            log_file.write("\n")

            for i in range(len(trpolearners)):
                trpo_scores[i] = trpo_scores[i] / score_sum
                trpoprobs[i] = trpoprobs[i] * 0.9 + trpo_scores[i]

            
            current_agent = -1
            max_score = 0.5 * maml_score / score_sum
            

            for i in range(len(trpolearners)):
                if trpoprobs[i] * trpo_scores[i] > max_score:
                    max_score = trpoprobs[i] * trpo_scores[i]
                    current_agent = i

                learning_rate = trpo_scores[i]
                maxd = 0.1 * trpo_scores[i] * trpo_scores[i]
                overfitted = False
                while True:
                    l, d = trpolearners[i].optimize_policy_batch(learning_rate, state_vector, action_vector,  reward_vector)
                    
                    #print("Epoch " + str(epoch) +  " Policy_lr: " + str(learning_rate) + " Loss:" + str(l) + " Divergence:" + str(d) )
                    if d > maxd:
                        overfitted = True
                        learning_rate /= 2.
                    else:
                        if overfitted:
                            break
                        else:
                            learning_rate *= 2.
                    if learning_rate > 100.:
                        break

                trpolearners[i].update_policy()
                trpolearners[i].update_next_state( state_vector, action_vector, nextstate_vector, learning_rate)

            if current_agent == -1:
                trpolearners.append(TRPOLearner(state_dim, action_dim, name=str(len(trpolearners))))
                trpolearners[-1].init_from_maml(mamllearner)
                trpoprobs.append(0.5)
                current_agent = len(trpolearners) - 1
                print("New Agent")

env.close()