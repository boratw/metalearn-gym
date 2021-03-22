
import numpy as np
class DummyEnv():
    

    def get_current_obs(self):
        return np.array([0., 0., 0., 0., 0., 0.])
    
    @property
    def action_space(self):
        return np.array([0., 0.])

    def reset(self):
        self.state = np.array([np.random.uniform(-0.1, 0.1),  np.random.uniform(-0.1, 0.1),  np.random.uniform(1.0, 3.0),  np.random.uniform(1.0, 3.0),  np.random.uniform(1.0, 3.0),  np.random.uniform(1.0, 3.0)])
        #self.state = np.array([np.random.uniform(-0.1, 0.1),  np.random.uniform(-0.1, 0.1),  1.0, 1.0])
        return self.state
    
    def step(self, action):
        self.state[0] += action[0] * 0.05
        self.state[1] += action[1] * 0.05

        reward =  (2.0 - (self.state[0] - (self.state[2] + self.state[4]) / 2 ) ** 2 - (self.state[1] - (self.state[3] + self.state[5]) / 2) ** 2) / 2.
        notdone = (-1.5 < self.state[0] < 3.5) and (-1.5 < self.state[1] < 3.5)
        done = not notdone
        return self.state, reward, done, np.array([])