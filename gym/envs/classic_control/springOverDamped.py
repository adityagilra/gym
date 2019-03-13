import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class SpringOverDampedEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_position=2.
        #self.max_force=np.finfo(np.float32).max
        self.max_force=1.
        self.t=0.
        self.dt=.001#.05
        self.viewer = None

        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force,
                                        shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.max_position, high=self.max_position,
                                        shape=(1,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        posn = self.state # posn := position

        dt = self.dt
        self.t += dt

        u = np.clip(u, -self.max_force, self.max_force)[0]
        self.last_u = u # for rendering
        #costs = (posn+0.5)**2 #+ .001*(u**2)
        if self.t>0.1:
            costs = (posn+0.5)**2 #+ .001*(u**2)
        else: costs = 0.

        newposn = posn + 5*u*dt

        self.state = newposn
        return self._get_obs(), -costs, False, {}

    def reset(self):
        self.t = 0.
        self.state = 0.
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return np.array((self.state,))

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(.1, .1)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_translation(self.state, 0.)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
