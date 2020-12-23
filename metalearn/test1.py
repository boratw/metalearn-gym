import gym
import numpy as np
import tensorflow as tf
import random


input_state = tf.placeholder(tf.float32, [None, 4])
input_action = tf.placeholder(tf.float32, [None, 3])
input_qvalue = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.truncated_normal([4, 40], stddev=0.1), trainable=True, name="w1")
b1 = tf.Variable(tf.zeros([40]), trainable=True, name="b1")

fc1 = tf.matmul(input_state, w1) + b1
fc1_relu = tf.nn.relu(fc1)

w2 = tf.Variable(tf.truncated_normal([40, 20], stddev=0.1), trainable=True, name="w2")
b2 = tf.Variable(tf.zeros([20]), trainable=True, name="b2")

fc2 = tf.matmul(fc1_relu, w2) + b2
fc2_relu = tf.nn.relu(fc2)

w3 = tf.Variable(tf.truncated_normal([20, 3], stddev=0.1), trainable=True, name="w3")
b3 = tf.Variable(tf.zeros([3]), trainable=True, name="b3")
output = tf.matmul(fc2_relu, w3) + b3

cost = tf.reduce_sum(tf.square(tf.reduce_sum(tf.multiply(input_action, output), axis=1, keepdims=True) - input_qvalue)) * 0.01
operation = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=0)



log_file = open("data/test1/log1.txt", "wt")
env = gym.make('CartPole-v0')

for epoch in range(1000):
    states = []
    actions = []
    qvalues = []
    steps = 0
    costs = 0
    for play in range(16):
        rewards = []
        maxvalues = []
        state = env.reset()
        step = 0
        for _ in range(300):
            states.append(state)
            if play == 0:
                env.render()
            o_output = sess.run(output, {input_state:[state]})
            action = np.argmax(o_output[0])
            maxvalues.append(o_output[0][action])

            r = random.randrange(12)
            if r <= 2:
                action = r

            action_onehot = [0., 0., 0.]
            action_onehot[action] = 1.
            actions.append(action_onehot)
            step += 1

            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
                
        print("Episode " + str(epoch) + " Step " + str(step))
        steps += step
        for i in range(1, step) :
            qvalues.append([rewards[i-1] + maxvalues[i]])
        qvalues.append([rewards[step-1]])
    
    for _ in range(32):
        dic = random.sample(range(len(states)), 16)
        dic_states = [states[i] for i in dic]
        dic_actions = [actions[i] for i in dic]
        dic_qvalues = [qvalues[i] for i in dic]
        _, o_cost = sess.run((operation, cost), {input_state:dic_states, input_action:dic_actions, input_qvalue:dic_qvalues })

        print("Episode " + str(epoch) + " Loss " + str(o_cost))
        costs += o_cost

    log_file.write("Episode\t" + str(epoch) + "\tStep\t" + str(steps / 16) + "\tCost\t" + str(costs / 32) + "\n")
    if epoch % 50 == 0:
        saver.save(sess, "data/test1/log1_" + str(epoch) + ".ckpt")

        
env.close()