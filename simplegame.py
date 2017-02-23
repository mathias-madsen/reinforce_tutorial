import numpy as np
from matplotlib import pyplot as plt

import gym
from gym import spaces

from ReinforceAgent import ReinforceAgent
from PolynomialPolicy import StickyDiscretePolicy


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


game = SimpleEnvironment()

sdim = game.observation_space.shape[0]
usize = game.action_space.n

policy = StickyDiscretePolicy(sdim=sdim, usize=usize, degree=0,
                              shuffleprob=0.0, repeatprob=0.0)

agent = ReinforceAgent(policy=policy)
agent.train(game, I=5, N=10, T=100, gamma=0.95)

