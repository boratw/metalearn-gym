import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Ant6Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant6.xml', 5)
        utils.EzPickle.__init__(self)



    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = 0.01 * np.square(a).sum()
        #contact_cost = 0.
        #moving_reward = 0.005 * np.sum(np.square(self.sim.data.qvel))
        contact_cost = 0.05 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.1
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.3 and state[2] <= 1.0
        reward = (forward_reward - ctrl_cost - contact_cost + survive_reward)
        done = not notdone
        ob = self.get_current_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.data.get_body_xmat("torso").flat
        ]).reshape(-1)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
