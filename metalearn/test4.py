import gym
import numpy as np
import tensorflow as tf
import random


tf.set_random_seed(1)

input_state = tf.placeholder(tf.float32, [None, 4])
input_action = tf.placeholder(tf.float32, [None, 3])
input_qvalue = tf.placeholder(tf.float32, [None, 1])
global_step = tf.placeholder(tf.int64)

w1 = tf.Variable(tf.truncated_normal([4, 32], stddev=0.1), trainable=True, name="w1")
b1 = tf.Variable(tf.truncated_normal([32], stddev=0.01), trainable=True, name="b1")

fc1 = tf.matmul(input_state, w1) + b1
fc1_act = tf.nn.leaky_relu(fc1, alpha=0.05)

w2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1), trainable=True, name="w2")
b2 = tf.Variable(tf.truncated_normal([32], stddev=0.01), trainable=True, name="b2")

fc2 = tf.matmul(fc1_act, w2) + b2
fc2_act = tf.nn.leaky_relu(fc2, alpha=0.05)

w3 = tf.Variable(tf.truncated_normal([32, 4], stddev=0.1), trainable=True, name="w3")
fc3 = tf.matmul(fc2_act, w3)

output_v, output_a = tf.split(fc3, [1, 3], 1)

cost = tf.reduce_sum(tf.clip_by_value(tf.square(output_v + tf.reduce_sum(tf.multiply(input_action, output_a), axis=1, keepdims=True) - input_qvalue), 0., 1024.)) * 0.001

learning_rate = tf.train.exponential_decay(0.001, global_step, 40, 0.9) 
operation = tf.train.AdamOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=0)


env = gym.make('CartPoleHead-v0')

log_file = open("data/test4_3/log1.txt", "wt")


for epoch in range(1001):
    states = []
    actions = []
    qvalues = []
    qadvs = []
    lastturn = []
    steps = 0
    costs = 0
    scores = 0
    for play in range(16):
        rewards = []
        maxvalues = []
        state = env.reset()
        env.set_goal(0.5)
        previousrandom = -1
        step = 0
        for _ in range(1000):
            states.append(state)
            if play == 0:
                env.render()
            o_v, o_a = sess.run((output_v, output_a), {input_state:[state]})
            action = np.argmax(o_a[0])
            maxvalues.append(o_v[0][0] + o_a[0][action])

            if play == 0:
                r = 99
            else:
                if previousrandom != -1 :
                    r = random.randrange(6)
                    if r < 5:
                        r = previousrandom
                    else:
                        r = 99
                else :
                    r = random.randrange(20 + int(epoch / 3) )
            if r <= 2:
                action = r
                previousrandom = r
            else:
                previousrandom = -1

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
            qvalues.append([rewards[i-1] + maxvalues[i] * 0.96])
        if step == 1000:
            states = states[:-1]
            actions = actions[:-1]
        else:
            qvalues.append([-25.])
        lastturn.append(len(qvalues) - 1)

    
    for play in range(64):
        if play == 0:
            dic = random.sample(range(len(states)), 48)
            dic.extend(lastturn)
        else:
            dic = random.sample(range(len(states)), 64)
        dic_states = [states[i] for i in dic]
        dic_actions = [actions[i] for i in dic]
        dic_qvalues = [qvalues[i] for i in dic]
        _, o_cost = sess.run((operation, cost), {input_state:dic_states, input_action:dic_actions, input_qvalue:dic_qvalues, global_step:epoch })

        costs += o_cost

    print("Episode " + str(epoch) + " Loss " + str(costs / 32))
    log_file.write("Episode\t" + str(epoch) + "\tStep\t" + str(steps / 16) +  "\tScore\t" + str(scores / 16) + "\tCost\t" + str(costs / 32) + "\n")
    if epoch % 50 == 0:
        saver.save(sess, "data/test4_3/log1_" + str(epoch) + ".ckpt")

        
env.close()