import gym
import random
import time
import numpy as np

action = 1

def key_press(key, mod):
    global action
    if key == ord('1'):
        action = 0
    if key == ord('2'):
        action = 1
    if key == ord('3'):
        action = 2



env = gym.make('AcrobotReach-v1')
env.render()
env.viewer.window.on_key_press = key_press

for epoch in range(1000):
    state = env.reset()
    env.set_goal([1.0, 0.5])
    env.render()
    env.viewer.window.on_key_press = key_press
    step = 0
    for _ in range(1000):
        env.render()
        step += 1

        state, reward, done, info = env.step(action)
        print(np.sin(state[0]) + np.sin(state[0] + state[1]), -np.cos(state[0]) - np.cos(state[0] + state[1]))
        if done:
            break
        time.sleep(0.2)
            
    print("Episode " + str(epoch) + " Step " + str(step))

env.close()