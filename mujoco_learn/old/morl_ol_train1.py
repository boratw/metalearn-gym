import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
import xml.etree.ElementTree as elemTree
from gym.envs.mujoco.ant6 import Ant6Env
from gym.envs.mujoco.ant6_modified import Ant6ModifiedEnv
from networks.morl_ol_learner import MORLLearner
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


tasks = [   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            [2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   ]

gamma = 0.96
horizon = 20
action_variance = 4
qvalue_variance = 1
action_maximum = 1.
running_samples = 200

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/morl_ol5/"
log_file = open(LOG_DIR + "log.txt", "wt")

mamllearner = MORLLearner(state_dim, action_dim, gamma=gamma, action_maximum=action_maximum, state_lr=0.01, qvalue_lr=0.01)
trpolearners = [MORLLearner(state_dim, action_dim, name="0", gamma=gamma, action_maximum=action_maximum, state_lr=0.005, qvalue_lr=0.005)]
maml_scores = [ 0.45 ]
trpo_scores = [ [0.55] ]
learners_len = 1
scores_record = 5

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] >= 0.27 and state[2] <= 1.0
    return not notdone

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    current_agent = 0
    cur_epoch_length = 0

    state_maml_vector = []
    action_maml_vector = []
    reward_maml_vector = []
    nextstate_maml_vector = []

    state_store_vector = []
    action_store_vector = []
    reward_store_vector = []
    nextstate_store_vector = []

    for (taski, task) in enumerate(tasks):
        SetGoal(task)
        env.close()
        env = Ant6ModifiedEnv()

        
        for epoch in range(1, 31):

            state_vector = []
            action_vector = []
            reward_vector = []
            nextstate_vector = []

            cur_epoch_length += 1
            data_len = 0
            state_corr = [0.] * (learners_len + 1)

            cur_state_vector = []
            cur_action_vector = []

        
            print("Epoch " + str(epoch) + " Start Collecting")
            while data_len < 500:
                logging = True
                curreward = 0.
                state = env.reset()
                for play in range(201):
                    prevstate = state

                    action = trpolearners[current_agent].get_action_optimal(state)
                    cur_state_vector.append(prevstate)
                    cur_action_vector.append(action)
                    state, reward, done, _ = env.step(action * 0.1)
                    curreward += reward

                    for i in range(learners_len):
                        state_corr[i] += trpolearners[i].get_next_diff(prevstate, action, state)
                    state_corr[learners_len] += mamllearner.get_next_diff(prevstate, action, state)

                    state_vector.append(prevstate)
                    action_vector.append(action)
                    nextstate_vector.append(state)
                    reward_vector.append([reward])
                    data_len += 1

                    if(is_done(state)):
                        break
                if logging:
                    log_reward = curreward
                    log_step = play
                    logging = False
        
            print("Epoch " + str(epoch) + " Reward: " + str(log_reward) + " Step:" + str(log_step))

            while data_len < 2000:
                state = env.reset()
                for play in range(201):
                    prevstate = state

                    action = trpolearners[current_agent].get_action_collecting(state)
                    state, reward, done, _ = env.step(action * 0.1)

                    for i in range(learners_len):
                        state_corr[i] += trpolearners[i].get_next_diff(prevstate, action, state)
                    state_corr[learners_len] += mamllearner.get_next_diff(prevstate, action, state)

                    state_vector.append(prevstate)
                    action_vector.append(action)
                    nextstate_vector.append(state)
                    reward_vector.append([reward])
                    data_len += 1

                    if(done):
                        break

            state_corr = [ (1. / (s + 0.001)) ** 2 for s in state_corr]
            state_corr = [ s / np.sum(state_corr) for s in state_corr]
            trpo_score = [0.] * learners_len
            for i in range(learners_len):
                trpo_scores[i].append(state_corr[i])
                if len(trpo_scores[i]) > scores_record:
                    trpo_scores[i] = trpo_scores[i][1:]
                trpo_score[i] = np.sum(trpo_scores[i]) ** 8
            maml_scores.append(state_corr[learners_len])
            if len(maml_scores) > scores_record:
                maml_scores = maml_scores[1:]
            maml_score = np.sum(maml_scores) ** 8

            score_sum = np.sum(trpo_score) + maml_score

            trpo_score /= score_sum
            maml_score /= score_sum

            print("Current Probability : ")
            print(state_corr)
            print("Learner Probability : ")
            print(trpo_score)
            print(maml_score)

            dic = random.sample(range(data_len), 20)
            state_maml_vector.extend([state_vector[i] for i in dic])
            action_maml_vector.extend([action_vector[i] for i in dic])
            reward_maml_vector.extend([reward_vector[i] for i in dic])
            nextstate_maml_vector.extend([nextstate_vector[i] for i in dic])

            dic = random.sample(range(data_len), 500)
            state_store_vector.extend([state_vector[i] for i in dic])
            action_store_vector.extend([action_vector[i] for i in dic])
            reward_store_vector.extend([reward_vector[i] for i in dic])
            nextstate_store_vector.extend([nextstate_vector[i] for i in dic])


            if len(state_maml_vector) > 2000:
                dic = random.sample(range(len(state_maml_vector)), 2000)
                state_maml_vector = [state_maml_vector[x] for x in dic]
                action_maml_vector = [action_maml_vector[x] for x in dic]
                reward_maml_vector = [reward_maml_vector[x] for x in dic]
                nextstate_maml_vector = [nextstate_maml_vector[x] for x in dic]

            if len(state_store_vector) > 2000:
                cut = len(state_store_vector) - 2000
                state_store_vector = state_store_vector[cut:]
                action_store_vector = action_store_vector[cut:]
                reward_store_vector = reward_store_vector[cut:]
                nextstate_store_vector = nextstate_store_vector[cut:]

            log_loss_state = [0.] * learners_len
            log_loss_reward = [0.] * learners_len
            log_loss_qvalue = [0.] * learners_len
            log_loss_policy = [0.] * learners_len



            print("Epoch " + str(epoch) + " Start State Training")
            for i in range(learners_len):
                log_loss_state[i], log_loss_reward[i] = trpolearners[i].optimize_nextstate_batch(trpo_score[i] * trpo_score[i],
                 state_store_vector, action_store_vector, reward_store_vector, nextstate_store_vector)
            log_meta_loss_state, log_meta_loss_reward = mamllearner.optimize_nextstate_batch(( (0.25 if taski == 0 else 1.) / (learners_len + 1) * (learners_len + 1)),
                state_maml_vector, action_maml_vector, reward_maml_vector, nextstate_maml_vector)
            
            print("Epoch " + str(epoch) + " State Loss : " + str(log_loss_state) + " Reward Loss : " + str(log_loss_reward))
            print("Epoch " + str(epoch) + " State Meta Loss : " + str(log_meta_loss_state) + " Reward Meta Loss : " + str(log_meta_loss_reward))

            print("Epoch " + str(epoch) + " Start Model Running")
            running_sample_n = [ int(s * 500) + 1 for s in trpo_score ]
            for m in range(learners_len):
                running_sample = running_sample_n[m]
                dic = random.sample(range(data_len), running_sample)
                policy_state_vector = []
                policy_action_vector = []
                policy_qvalue_vector = []
                first_state = [state_vector[i] for i in dic]
                for i in range(action_variance):

                    first_action = trpolearners[m].get_actions_diverse(first_state)
                    currewards = []
                    for j in range(qvalue_variance):
                        finished_vector = [False] * running_sample
                        survive_vector = np.ones(running_sample) * gamma
                        state, reward = trpolearners[m].get_nexts(first_state, first_action)
                        curreward = reward
                        for j in range(horizon):
                            for k in range(running_sample):
                                if is_done(state[k]) and not finished_vector[k]:
                                    finished_vector[k] = True
                                    survive_vector[k] = 0.
                                    curreward[k] -= 1. * gamma ** j
                            state, reward = trpolearners[m].get_nexts_with_policy(state)
                            curreward = np.add(curreward, np.multiply(reward, survive_vector))
                            survive_vector *= gamma
                        qvalue = trpolearners[m].get_qvalues_with_policy(state)
                        curreward = np.add(curreward, np.multiply(qvalue, survive_vector * 0.5))
                        currewards.append(curreward)

                    maxreward = np.amax(currewards, axis=0)
                        
                    policy_state_vector.extend(first_state)
                    policy_action_vector.extend(first_action)
                    policy_qvalue_vector.extend(maxreward)

                policy_qvalue_vector = np.reshape(policy_qvalue_vector, (-1, 1))

                print("Epoch " + str(epoch) + " Learner " + str(m) + " Start Qvalue Training")
                log_loss_qvalue[m] = trpolearners[m].optimize_qvalue_batch(trpo_score[m] * trpo_score[m], policy_state_vector, policy_action_vector, policy_qvalue_vector)
                print("Epoch " + str(epoch) + " Learner " + str(m) + " Qvalue Training Loss : " + str(log_loss_qvalue[m]))
                print("Epoch " + str(epoch) + " Learner " + str(m) + " Start Policy Training")
                log_loss_policy[m] = trpolearners[m].optimize_policy_batch(trpo_score[m] * trpo_score[m], cur_state_vector)
                print("Epoch " + str(epoch) + " Learner " + str(m) + " Policy Training Loss : " + str(log_loss_policy[m]))
                
                    
            running_sample = 500
            dic = random.sample(range(data_len), running_sample)
            policy_state_vector = []
            policy_action_vector = []
            policy_qvalue_vector = []
            first_state = [state_vector[i] for i in dic]
            for i in range(action_variance):

                first_action = mamllearner.get_actions_diverse(first_state)
                currewards = []
                for j in range(qvalue_variance):
                    finished_vector = [False] * running_sample
                    survive_vector = np.ones(running_sample) * gamma
                    state, reward = mamllearner.get_nexts(first_state, first_action)
                    curreward = reward
                    for j in range(horizon):
                        for k in range(running_sample):
                            if is_done(state[k]) and not finished_vector[k]:
                                finished_vector[k] = True
                                survive_vector[k] = 0.
                                curreward[k] -= 1. * gamma ** j
                        state, reward = mamllearner.get_nexts_with_policy(state)
                        curreward = np.add(curreward, np.multiply(reward, survive_vector))
                        survive_vector *= gamma
                    qvalue = mamllearner.get_qvalues_with_policy(state)
                    curreward = np.add(curreward, np.multiply(qvalue, survive_vector * 0.5))
                    currewards.append(curreward)

                maxreward = np.amax(currewards, axis=0)
                    
                policy_state_vector.extend(first_state)
                policy_action_vector.extend(first_action)
                policy_qvalue_vector.extend(maxreward)

            policy_qvalue_vector = np.reshape(policy_qvalue_vector, (-1, 1))

            print("Epoch " + str(epoch) + " Learner Meta Start Qvalue Training")
            log_meta_loss_qvalue = mamllearner.optimize_qvalue_batch(((0.25 if taski == 0 else 1.) / (learners_len + 1) * (learners_len + 1)), policy_state_vector, policy_action_vector, policy_qvalue_vector)
            print("Epoch " + str(epoch) + " Learner Meta Qvalue Training Loss : " + str(log_loss_qvalue[m]))
            print("Epoch " + str(epoch) + " Learner Meta Start Policy Training")
            log_meta_loss_policy = mamllearner.optimize_policy_batch(((0.25 if taski == 0 else 1.) / (learners_len + 1) * (learners_len + 1)), cur_state_vector)
            print("Epoch " + str(epoch) + " Learner Meta Policy Training Loss : " + str(log_loss_policy[m]))


            log_file.write(
                "\tReward\t" + str(log_reward) +
                "\tSteps\t" + str(log_step) +
                "\tProbability Meta\t" + str(maml_score) +
                "\tLossState Meta\t" + str(log_meta_loss_state) +
                "\tLossReward Meta\t" + str(log_meta_loss_reward) +
                "\tLossQvalue Meta\t" + str(log_meta_loss_qvalue) +
                "\tLossPolicy Meta\t" + str(log_meta_loss_policy))
            for m in range(learners_len):
                log_file.write(
                    "\tProbability " + str(m) + "\t" + str(trpo_score[m]) +
                    "\tLossState " + str(m) + "\t" + str(log_loss_state[m]) +
                    "\tLossReward " + str(m) + "\t" + str(log_loss_reward[m]) +
                    "\tLossQvalue " + str(m) + "\t" + str(log_loss_qvalue[m]) +
                    "\tLossPolicy " + str(m) + "\t" + str(log_loss_policy[m]))
            log_file.write("\n")


            if cur_epoch_length % 100 == 0:
                saver.save(sess, LOG_DIR + "log_" + str(cur_epoch_length) + ".ckpt")

            if maml_score > np.max(trpo_score) :
                current_agent = learners_len
            else:
                current_agent = np.argmax(trpo_score)
            if current_agent == learners_len:
                trpolearners.append(MORLLearner(state_dim, action_dim, name=str(len(trpolearners)), gamma=gamma, action_maximum=action_maximum, state_lr=0.005, qvalue_lr=0.005))
                trpolearners[-1].init_from_maml(mamllearner)

                trpo_scores.append(list(maml_scores))
                trpo_scores[-1][-1] *= 2
                learners_len += 1
                print("!!!New Agent!!!")

env.close()