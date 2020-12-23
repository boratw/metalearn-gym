import gym
import numpy as np
import tensorflow as tf
import random


input_state = tf.placeholder(tf.float32, [None, 4])
input_action = tf.placeholder(tf.float32, [None, 5])
input_qvalue = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.zeros([4, 20], stddev=0.1), trainable=True, name="w1")
b1 = tf.Variable(tf.zeros([20]), trainable=True, name="b1")

fc1 = tf.matmul(input_state, w1) + b1
fc1_relu = tf.nn.relu(fc1)

w2 = tf.Variable(tf.zeros([20, 20], stddev=0.1), trainable=True, name="w2")
b2 = tf.Variable(tf.zeros([20]), trainable=True, name="b2")

fc2 = tf.matmul(fc1_relu, w2) + b2
fc2_relu = tf.nn.relu(fc2)

w3 = tf.Variable(tf.zeros([20, 5], stddev=0.1), trainable=True, name="w3")
b3 = tf.Variable(tf.zeros([5]), trainable=True, name="b3")
output = tf.matmul(fc2_relu, w3) + b3

cost = tf.reduce_sum(tf.square(tf.reduce_sum(tf.multiply(input_action, output), axis=1, keepdims=True) - input_qvalue))
operation = tf.train.AdamOptimizer(0.01).minimize(cost)



input_w1 = tf.placeholder(tf.float32, [None, 4, 20])
input_b1 = tf.placeholder(tf.float32, [None, 20])
input_w2 = tf.placeholder(tf.float32, [None, 20, 20])
input_b2 = tf.placeholder(tf.float32, [None, 20])
input_w3 = tf.placeholder(tf.float32, [None, 20, 5])
input_b3 = tf.placeholder(tf.float32, [None, 5])


w1_phi = tf.Variable(tf.truncated_normal([4, 20], stddev=0.1), trainable=True, name="w1")
b1_phi = tf.Variable(tf.zeros([20]), trainable=True, name="b1")
w2_phi = tf.Variable(tf.truncated_normal([4, 20], stddev=0.1), trainable=True, name="w1")
b2_phi = tf.Variable(tf.zeros([20]), trainable=True, name="b1")
w3_phi = tf.Variable(tf.truncated_normal([4, 20], stddev=0.1), trainable=True, name="w1")
b3_phi = tf.Variable(tf.zeros([20]), trainable=True, name="b1")


cost_meta = tf.reduce_sum(tf.square(input_w1 - w1_phi)) + \
        tf.reduce_sum(tf.square(input_w2 - w2_phi)) + \
        tf.reduce_sum(tf.square(input_w3 - w3_phi)) + \
        tf.reduce_sum(tf.square(input_b1 - b1_phi)) + \
        tf.reduce_sum(tf.square(input_b2 - b2_phi)) + \
        tf.reduce_sum(tf.square(input_b3 - b3_phi))


operation_meta = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep=0)


env = gym.make('CartPole-v0')

for epoch in range(1000):
    states = []
    actions = []
    qvalues = []
    for _ in range(8):
        state = env.reset()
        step = 0
        for _ in range(200):
            states.append(state)
            env.render()
            o_output = sess.run(output, {input_state:[state]})
            action = np.argmax(o_output[0])

            r = random.randrange(16)
            if r <= 4:
                action = r

            action_onehot = [0., 0., 0., 0., 0.]
            action_onehot[action] = 1.
            actions.append(action_onehot)
            step += 1

            state, reward, done, info = env.step(action)
            if done:
                break
                
        print("Episode " + str(epoch) + " Step " + str(step))
        rewards = []
        reward = 0.
        for i in range(step) :
            reward += 0.8 ** i
            rewards.append([reward])
        qvalues.extend(rewards[::-1])
    
    for _ in range(32):
        dic = random.sample(range(len(states)), 16)
        dic_states = [states[i] for i in dic]
        dic_actions = [actions[i] for i in dic]
        dic_qvalues = [qvalues[i] for i in dic]
        _, o_cost = sess.run((operation, cost), {input_state:dic_states, input_action:dic_actions, input_qvalue:dic_qvalues })

        print("Episode " + str(epoch) + " Loss " + str(o_cost))

    if epoch % 10 == 0:
        saver.save(sess, "data/test1/log1_" + str(epoch) + ".ckpt")

        
env.close()