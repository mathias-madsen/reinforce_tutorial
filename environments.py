import numpy as np
from matplotlib import pyplot as plt

from gym import spaces


class SimpleEnvironment(object):
    
    def __init__(self, sdim=3, usize=2):
        
        self.sdim = sdim
        self.usize = usize
        
        self.action_space = spaces.discrete.Discrete(usize)
        self.observation_space = spaces.box.Box(np.zeros(sdim), np.ones(sdim))
        
        self.state = self.observation_space.sample()
    
    def reset(self):
        
        self.state = self.observation_space.sample()
        
        return self.state
    
    def dynamics(self, state, action):
        
        return self.observation_space.sample()
    
    def step(self, action):
        
        self.state = self.dynamics(self.state, action)
        reward = 1.0 if action == 0 else -1.0 # switch statement

        return self.state, reward, False, {} # obs, reward, done, oddities

