import gym
import numpy as np
import tensorflow as tf
import random
import gym.envs.mujoco  
import time
from gym.envs.mujoco.ant6_modified import Ant6ModifiedEnv
import xml.etree.ElementTree as elemTree
import moviepy.editor as mpy

from networks.sac_learner2 import SACLearner

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetGoal(goal, name) :
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


    tree.write("../gym/envs/mujoco/assets/" + name)

SetGoal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "ant6_modified.xml")

env = Ant6ModifiedEnv()


tasks = [   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.01, 1.0, 1.0], 
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0], 
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.01, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            [0.01, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.01, 1.0, 1.0, 0.01, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.01, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] ,
            [1.0, 1.0, 0.01, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]    ]

state_dim = env.get_current_obs().size
action_dim = env.action_space.shape[0]
gamma = 0.96

print("state_dim", state_dim)
print("action_dim", action_dim)

LOG_DIR = "data/sac_ol3/"
log_file = open(LOG_DIR + "log.txt", "wt")

learners = [SACLearner(state_dim, action_dim, name="0", value_lr=0.0005, qvalue_lr=0.0005, policy_lr=0.0005, gamma=gamma)]
agent_finding = [False]
learner_len = 1
baseline_score = 1.

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    learners[0].value_network_initialize()

    vector_len = 0
    state_vector = []
    next_state_vector = []
    action_vector = []
    reward_vector = []
    survive_vector = []
    baseline_over = 0

    cur_agent = 0

    for (taski, task) in enumerate(tasks):
        SetGoal(task, "ant6_modified.xml")
        #env.close()
        env = Ant6ModifiedEnv()

        for epoch in range(2001):
            if epoch % 400 == 0:
                record_image = True
            else:
                record_image = False
            images = []

            print("Task " + str(taski) + " Epoch " + str(epoch))
            totalreward = 0.
            state_score = np.zeros((learner_len))
            for play in range(8):
                state = env.reset()
                curreward = 0.
                cur_state_score = np.zeros((learner_len))
                for step in range(200):
                    prev_state = state

                    action = learners[cur_agent].get_action_stochastic(state)[0]
                    state, reward, done, _ = env.step(action)
                    for agent in range(learner_len):
                        state_est = learners[agent].get_next_state(prev_state, action)
                        cur_state_score[agent] += np.mean((state - state_est) ** 2)
                    
                    state_vector.append(prev_state)
                    action_vector.append(action)
                    survive_vector.append([ 0.0 if done else 1.0])
                    next_state_vector.append(state)
                    reward_vector.append([reward])

                    vector_len += 1
                    if record_image and play < 4:
                        env.render()
                        image = env.viewer.read_pixels(2500, 1400, False)
                        #pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                        #images.append(np.flipud(np.array(pil_image)))
                        images.append(np.flipud(np.array(image)))
                    if(done):
                        break
                    curreward += reward
                state_score += cur_state_score / (step + 1)
                print("Agent : " + str(cur_agent) + " Reward : " + str(curreward))
                totalreward += curreward
            totalreward /= 8
            state_score /= 8
            for agent in range(learner_len):
                print("State Score " + str(agent) + " : " + str(state_score[agent]))
            print("Baseline Score : " + str(baseline_score))
        


            qs = 0.
            vs = 0.
            ps = 0.
            for history in range(32):
                dic = random.sample(range(vector_len), 256 if vector_len > 256 else vector_len)

                state_vector_dic = [state_vector[x] for x in dic]
                next_state_vector_dic = [next_state_vector[x] for x in dic]
                action_vector_dic = [action_vector[x] for x in dic]
                reward_vector_dic = [reward_vector[x] for x in dic]
                survive_vector_dic = [survive_vector[x] for x in dic]

                q, v, p, _, _, _, _ = learners[cur_agent].optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

                qs += np.mean(q)
                vs += np.mean(v)
                ps += np.mean(p)

            learners[cur_agent].value_network_update()
            qs /= 32.
            vs /= 32.
            ps /= 32.

            vec_trunc = vector_len // 10
            state_vector = state_vector[vec_trunc:]
            next_state_vector = next_state_vector[vec_trunc:]
            action_vector = action_vector[vec_trunc:]
            reward_vector = reward_vector[vec_trunc:]
            survive_vector = survive_vector[vec_trunc:]
            vector_len -= vec_trunc


            log_file.write(str(taski) + "\t" + str(epoch) + "\t" + str(cur_agent) + "\t" + str(totalreward) + "\tLearner\t" + str(cur_agent) + "\t" + str(qs) + "\t" + str(vs) + "\t" + str(ps) + "\t" + str(baseline_score))
            for s in state_score:
                log_file.write("\t" + str(s))
            log_file.write("\n")
            

            maximum_learner = np.argmin(state_score)
            
            if baseline_score > state_score[maximum_learner]:
                baseline_score = (baseline_score * 0.95) + (state_score[maximum_learner] * 0.075)

            if baseline_score < state_score[cur_agent]:
                baseline_over += 1
                if baseline_over > 3:
                    baseline_over = 3
            else:
                baseline_over -= 1
                if baseline_over < 0:
                    baseline_over = 0
                    agent_finding = [False] * learner_len
            
            if baseline_over >= 3:
                baseline_over = 0
                agent_finding[cur_agent] = True
                cur_agent = -1
                min_value = 99999.
                for agent in range(learner_len):
                    if agent_finding[agent] == False and state_score[agent] < min_value:
                        cur_agent = agent
                        min_value = state_score[agent]
                if cur_agent == -1:
                    learners.append( SACLearner(state_dim, action_dim, name=str(learner_len), value_lr=0.0005, qvalue_lr=0.0005, policy_lr=0.0005, gamma=gamma) )
                    cur_agent = learner_len
                    learner_len += 1
                    learners[-1].init_from_other(learners[maximum_learner])
                    agent_finding = [False] * learner_len

                    baseline_score = state_score[maximum_learner] * 1.5


            if epoch == 2000:
                saver.save(sess, LOG_DIR + "log_Task_" + str(taski) + "_Epoch_" + str(epoch) + ".ckpt")

            if record_image:
                clip = mpy.ImageSequenceClip(images, fps=30)
                clip.write_videofile(LOG_DIR + "Task_" + str(taski) + "_Epoch_" + str(epoch) + ".mp4", fps=30)
                env.close()