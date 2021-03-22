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

LOG_DIR = "data/meta_sac2/"
log_file = open(LOG_DIR + "log.txt", "wt")

def is_done(state):
    notdone = np.isfinite(state).all() and state[2] <= 1.0 and state[2] >= 0.27
    return not notdone

learner1 = SACLearner(state_dim, action_dim, name="1", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)
learner2 = SACLearner(state_dim, action_dim, name="2", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)
learner3 = SACLearner(state_dim, action_dim, name="3", value_lr=0.0001, qvalue_lr=0.0001, policy_lr=0.0001, gamma=gamma)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=0)

log_file.write("Episode\t" +
    "Env1_Score\tEnv1_Qvalue\tEnv1_Value\tEnv1_Policy\tEnv1_Qvalue_Loss\tEnv1_Value_Loss\tEnv1_Policy_Loss\tEnv1_State_Loss\t" +
    "Env2_Score\tEnv2_Qvalue\tEnv2_Value\tEnv2_Policy\tEnv2_Qvalue_Loss\tEnv2_Value_Loss\tEnv2_Policy_Loss\tEnv2_State_Loss\t" +
    "Env3_Score\tEnv3_Qvalue\tEnv3_Value\tEnv3_Policy\tEnv3_Qvalue_Loss\tEnv3_Value_Loss\tEnv3_Policy_Loss\tEnv3_State_Loss\n")

with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)
    learner1.value_network_initialize()
    learner2.value_network_initialize()
    learner3.value_network_initialize()

    vector_len1 = 0
    state_vector1 = []
    next_state_vector1 = []
    action_vector1 = []
    reward_vector1 = []
    survive_vector1 = []

    vector_len2 = 0
    state_vector2 = []
    next_state_vector2 = []
    action_vector2 = []
    reward_vector2 = []
    survive_vector2 = []

    vector_len3 = 0
    state_vector3 = []
    next_state_vector3 = []
    action_vector3 = []
    reward_vector3 = []
    survive_vector3 = []

    for epoch in range(100):
        state = env1.reset()
        for play in range(200):
            prev_state = state
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env1.step(action)
            state_vector1.append(prev_state)
            action_vector1.append(action)
            survive_vector1.append([ 0.0 if done else 1.0])
            next_state_vector1.append(state)
            reward_vector1.append([reward])

            vector_len1 += 1
            if(done):
                break

        state = env2.reset()
        for play in range(200):
            prev_state = state
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env2.step(action)
            state_vector2.append(prev_state)
            action_vector2.append(action)
            survive_vector2.append([ 0.0 if done else 1.0])
            next_state_vector2.append(state)
            reward_vector2.append([reward])

            vector_len2 += 1
            if(done):
                break

        state = env3.reset()
        for play in range(200):
            prev_state = state
            action = np.random.uniform(-1., 1., (action_dim))
            state, reward, done, _ = env3.step(action)
            state_vector3.append(prev_state)
            action_vector3.append(action)
            survive_vector3.append([ 0.0 if done else 1.0])
            next_state_vector3.append(state)
            reward_vector3.append([reward])

            vector_len3 += 1
            if(done):
                break
                

        print("InitialEpoch : " + str(epoch))

    for epoch in range(1, 50001):
        curreward1 = 0.
        state = env1.reset()
        for play in range(200):
            prev_state = state

            action = learner1.get_action_stochastic(state)[0]
            state, reward, done, _ = env1.step(action)
            
            state_vector1.append(prev_state)
            action_vector1.append(action)
            survive_vector1.append([ 0.0 if done else 1.0])
            next_state_vector1.append(state)
            reward_vector1.append([reward])

            vector_len1 += 1
            if(done):
                break
            curreward1 += reward

        print("Reward1 : " + str(curreward1))

        curreward2 = 0.
        state = env2.reset()
        for play in range(200):
            prev_state = state

            action = learner2.get_action_stochastic(state)[0]
            state, reward, done, _ = env2.step(action)
            
            state_vector2.append(prev_state)
            action_vector2.append(action)
            survive_vector2.append([ 0.0 if done else 1.0])
            next_state_vector2.append(state)
            reward_vector2.append([reward])

            vector_len2 += 1
            if(done):
                break
            curreward2 += reward

        print("Reward2 : " + str(curreward2))

        curreward3 = 0.
        state = env3.reset()
        for play in range(200):
            prev_state = state

            action = learner3.get_action_stochastic(state)[0]
            state, reward, done, _ = env3.step(action)
            
            state_vector3.append(prev_state)
            action_vector3.append(action)
            survive_vector3.append([ 0.0 if done else 1.0])
            next_state_vector3.append(state)
            reward_vector3.append([reward])

            vector_len3 += 1
            if(done):
                break
            curreward3 += reward

        print("Reward3 : " + str(curreward3))

        qs1 = 0.
        vs1 = 0.
        ps1 = 0.
        ql1 = 0.
        vl1 = 0.
        pl1 = 0.
        sl1 = 0.
        for history in range(8):
            dic = random.sample(range(vector_len1), 200)

            state_vector_dic = [state_vector1[x] for x in dic]
            next_state_vector_dic = [next_state_vector1[x] for x in dic]
            action_vector_dic = [action_vector1[x] for x in dic]
            reward_vector_dic = [reward_vector1[x] for x in dic]
            survive_vector_dic = [survive_vector1[x] for x in dic]

            q, v, p, ql, vl, pl, sl = learner1.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs1 += np.mean(q)
            vs1 += np.mean(v)
            ps1 += np.mean(p)
            ql1 += np.mean(ql)
            vl1 += np.mean(vl)
            pl1 += np.mean(pl)
            sl1 += np.mean(sl)
        learner1.value_network_update()
        qs1 /= 8.
        vs1 /= 8.
        ps1 /= 8.
        ql1 /= 8.
        vl1 /= 8.
        pl1 /= 8.
        sl1 /= 8.
        print("Epoch " + str(epoch) + " Learner 1 Mean Qvalue Training : " + str(qs1))
        print("Epoch " + str(epoch) + " Learner 1 Mean Value Training : " + str(vs1))
        print("Epoch " + str(epoch) + " Learner 1 Mean Policy Training : " + str(ps1))

        qs2 = 0.
        vs2 = 0.
        ps2 = 0.
        ql2 = 0.
        vl2 = 0.
        pl2 = 0.
        sl2 = 0.
        for history in range(8):
            dic = random.sample(range(vector_len2), 200)

            state_vector_dic = [state_vector2[x] for x in dic]
            next_state_vector_dic = [next_state_vector2[x] for x in dic]
            action_vector_dic = [action_vector2[x] for x in dic]
            reward_vector_dic = [reward_vector2[x] for x in dic]
            survive_vector_dic = [survive_vector2[x] for x in dic]

            q, v, p, ql, vl, pl, sl = learner2.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs2 += np.mean(q)
            vs2 += np.mean(v)
            ps2 += np.mean(p)
            ql2 += np.mean(ql)
            vl2 += np.mean(vl)
            pl2 += np.mean(pl)
            sl2 += np.mean(sl)
        learner2.value_network_update()
        qs2 /= 8.
        vs2 /= 8.
        ps2 /= 8.
        ql2 /= 8.
        vl2 /= 8.
        pl2 /= 8.
        sl2 /= 8.
        print("Epoch " + str(epoch) + " Learner 2 Mean Qvalue Training : " + str(qs2))
        print("Epoch " + str(epoch) + " Learner 2 Mean Value Training : " + str(vs2))
        print("Epoch " + str(epoch) + " Learner 2 Mean Policy Training : " + str(ps2))


        qs3 = 0.
        vs3 = 0.
        ps3 = 0.
        ql3 = 0.
        vl3 = 0.
        pl3 = 0.
        sl3 = 0.
        for history in range(8):
            dic = random.sample(range(vector_len3), 200)

            state_vector_dic = [state_vector3[x] for x in dic]
            next_state_vector_dic = [next_state_vector3[x] for x in dic]
            action_vector_dic = [action_vector3[x] for x in dic]
            reward_vector_dic = [reward_vector3[x] for x in dic]
            survive_vector_dic = [survive_vector3[x] for x in dic]

            q, v, p, ql, vl, pl, sl = learner3.optimize_batch(state_vector_dic, next_state_vector_dic, action_vector_dic, reward_vector_dic, survive_vector_dic)

            qs3 += np.mean(q)
            vs3 += np.mean(v)
            ps3 += np.mean(p)
            ql3 += np.mean(ql)
            vl3 += np.mean(vl)
            pl3 += np.mean(pl)
            sl3 += np.mean(sl)
        learner3.value_network_update()
        qs3 /= 8.
        vs3 /= 8.
        ps3 /= 8.
        ql3 /= 8.
        vl3 /= 8.
        pl3 /= 8.
        sl3 /= 8.
        print("Epoch " + str(epoch) + " Learner 3 Mean Qvalue Training : " + str(qs3))
        print("Epoch " + str(epoch) + " Learner 3 Mean Value Training : " + str(vs3))
        print("Epoch " + str(epoch) + " Learner 3 Mean Policy Training : " + str(ps3))


        vec_trunc = vector_len1 // 50
        state_vector1 = state_vector1[vec_trunc:]
        next_state_vector1 = next_state_vector1[vec_trunc:]
        action_vector1 = action_vector1[vec_trunc:]
        reward_vector1 = reward_vector1[vec_trunc:]
        survive_vector1 = survive_vector1[vec_trunc:]
        value_vector1 = value_vector1[vec_trunc:]
        vector_len1 -= vec_trunc

        vec_trunc = vector_len2 // 50
        state_vector2 = state_vector2[vec_trunc:]
        next_state_vector2 = next_state_vector2[vec_trunc:]
        action_vector2 = action_vector2[vec_trunc:]
        reward_vector2 = reward_vector2[vec_trunc:]
        survive_vector2 = survive_vector2[vec_trunc:]
        value_vector2 = value_vector2[vec_trunc:]
        vector_len2 -= vec_trunc

        vec_trunc = vector_len3 // 50
        state_vector3 = state_vector3[vec_trunc:]
        next_state_vector3 = next_state_vector3[vec_trunc:]
        action_vector3 = action_vector3[vec_trunc:]
        reward_vector3 = reward_vector3[vec_trunc:]
        survive_vector3 = survive_vector3[vec_trunc:]
        value_vector3 = value_vector3[vec_trunc:]
        vector_len3 -= vec_trunc

        log_file.write(str(epoch) + "\t" +
            str(curreward1) + "\t" + str(qs1) + "\t" + str(vs1) + "\t" + str(ps1) + "\t" + str(ql1) + "\t" + str(vl1) + "\t" + str(pl1) + "\t" + str(sl1) + "\t"+ 
            str(curreward2) + "\t" + str(qs2) + "\t" + str(vs2) + "\t" + str(ps2) + "\t" + str(ql2) + "\t" + str(vl2) + "\t" + str(pl2) + "\t" + str(sl2) + "\t" + 
            str(curreward3) + "\t" + str(qs3) + "\t" + str(vs3) + "\t" + str(ps2) + "\t" + str(ql3) + "\t" + str(vl3) + "\t" + str(pl3) + "\t" + str(sl3) + "\n")
        if epoch % 1000 == 0:
            saver.save(sess, LOG_DIR + "log_" + str(epoch) + ".ckpt")


env.close()