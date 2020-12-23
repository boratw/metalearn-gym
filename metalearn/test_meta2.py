import gym
import numpy as np
import tensorflow as tf
import random


tf.set_random_seed(1)


meta_input_state =  tf.placeholder(tf.float32, [None, 2])
meta_groundtruth =  tf.placeholder(tf.float32, [None, 1408])

meta_w1 = tf.Variable(tf.truncated_normal([2, 64], stddev=0.1), trainable=True)
meta_b1 = tf.Variable(tf.truncated_normal([64], stddev=0.01), trainable=True)
meta_fc1 = tf.matmul(meta_input_state, meta_w1) + meta_b1

meta_w2 = tf.Variable(tf.truncated_normal([64, 1408], stddev=0.1), trainable=True)
meta_b2 = tf.Variable(tf.truncated_normal([1408], stddev=0.01), trainable=True)
meta_output = tf.matmul(meta_fc1, meta_w2) + meta_b2

meta_cost = tf.reduce_sum(tf.square(meta_output - meta_groundtruth)) * 0.1
meta_operation = tf.train.AdamOptimizer(0.001).minimize(meta_cost)



agent_input_state = tf.placeholder(tf.float32, [None, 6])
agent_input_action = tf.placeholder(tf.float32, [None, 3])
agent_input_qvalue = tf.placeholder(tf.float32, [None, 1])

agent_w1 = tf.Variable(tf.truncated_normal([6, 32], stddev=0.1), trainable=True, name="w1")
agent_b1 = tf.Variable(tf.truncated_normal([32], stddev=0.01), trainable=True, name="b1")

agent_fc1 = tf.matmul(agent_input_state, agent_w1) + agent_b1
agent_fc1_act = tf.nn.leaky_relu(agent_fc1, alpha=0.05)

agent_w2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1), trainable=True, name="w2")
agent_b2 = tf.Variable(tf.truncated_normal([32], stddev=0.01), trainable=True, name="b2")

agent_fc2 = tf.matmul(agent_fc1_act, agent_w2) + agent_b2
agent_fc2_act = tf.nn.leaky_relu(agent_fc2, alpha=0.05)

agent_w3 = tf.Variable(tf.truncated_normal([32, 4], stddev=0.1), trainable=True, name="w3")
agent_fc3 = tf.matmul(agent_fc2_act, agent_w3)

agent_output_v, agent_output_a = tf.split(agent_fc3, [1, 3], 1)

agent_cost = tf.reduce_sum(tf.clip_by_value(tf.square(agent_output_v + tf.reduce_sum(tf.multiply(agent_input_action, agent_output_a), axis=1, keepdims=True) - agent_input_qvalue), 0., 1024.)) * 0.01

agent_operation = tf.train.AdamOptimizer(0.001).minimize(agent_cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=0)


env = gym.make('AcrobotReach-v1')

log_file = open("data/test_meta2_1/log1.txt", "wt")


for epoch in range(2001):
    goals = [[np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)] for _ in range(16)]
    o_param = sess.run(meta_output, {meta_input_state:goals})
    g_param = []

    final_steps = 0

    for envn in range(16):
        agent_w1.assign(np.reshape(o_param[envn][:192], (6, 32)))
        agent_b1.assign(o_param[envn][192:224])
        agent_w2.assign(np.reshape(o_param[envn][224:1248], (32, 32)))
        agent_b2.assign(o_param[envn][1248:1280])
        agent_w3.assign(np.reshape(o_param[envn][1280:], (32, 4)))

        states = []
        actions = []
        qvalues = []
        qadvs = []
        lastturn = []
        steps = 0
        costs = 0
        scores = 0
        for play in range(8):
            rewards = []
            maxvalues = []
            state = env.reset()
            env.set_goal(goals[envn])
            step = 0
            for _ in range(500):
                states.append(state)
                if envn == 0 and play == 0:
                    env.render()
                o_v, o_a = sess.run((agent_output_v, agent_output_a), {agent_input_state:[state]})
                action = np.argmax(o_a[0])
                maxvalues.append(o_v[0][0] + o_a[0][action])

                r = random.randrange(12)
                if r <= 2:
                    action = r

                action_onehot = [0., 0., 0.]
                action_onehot[action] = 1.
                actions.append(action_onehot)
                step += 1

                state, reward, done, info = env.step(action)
                rewards.append(reward)
                scores += reward
                if done:
                    break
                    
            print("Episode " + str(epoch) + " Step " + str(step))
            steps += step
            survivescore = step * 0.1
            for i in range(1, step) :
                qvalues.append([rewards[i-1] + maxvalues[i] * 0.95])
            if step == 500:
                states = states[:-1]
                actions = actions[:-1]
            else:
                qvalues.append([25.])
            lastturn.append(len(qvalues) - 1)

        
        for play in range(32):
            if play == 0:
                dic = random.sample(range(len(states)), 48)
                dic.extend(lastturn)
            else:
                dic = random.sample(range(len(states)), 64)
            dic_states = [states[i] for i in dic]
            dic_actions = [actions[i] for i in dic]
            dic_qvalues = [qvalues[i] for i in dic]
            _, o_cost = sess.run((agent_operation, agent_cost), {agent_input_state:dic_states, agent_input_action:dic_actions, agent_input_qvalue:dic_qvalues })

            costs += o_cost

        n_w1, n_b1, n_w2, n_b2, n_w3 = sess.run((agent_w1, agent_b1, agent_w2, agent_b2, agent_w3))
        n_param = np.concatenate((n_w1.flatten(), n_b1, n_w2.flatten(), n_b2, n_w3.flatten()))


        g_param.append(np.add(o_param[envn], n_param))

        print("Env " + str(envn) + " Episode " + str(epoch) + " Loss " + str(costs / 32) + " Step " + str(steps / 16))
        final_steps += steps / 16
        
    _, o_cost = sess.run((meta_operation, meta_cost), {meta_input_state:goals, meta_groundtruth:g_param})

    print("Episode " + str(epoch) + " Loss " + str(o_cost) + " Step " + str(final_steps / 16))
    log_file.write("Episode\t" + str(epoch) + "\tStep\t" + str(final_steps / 16) + "\tCost\t" + str(o_cost) + "\n")
    
    if epoch % 50 == 0:
        saver.save(sess, "data/test_meta2_1/log1_" + str(epoch) + ".ckpt")

        
env.close()