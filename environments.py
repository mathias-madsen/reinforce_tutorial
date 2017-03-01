import numpy as np
from matplotlib import pyplot as plt

from gym import spaces


class TargetPractice(object):
    
    def __init__(self, sdim=3, udim=2, goal=None):
        """ A game in which the goal is to perform a specific fixed action. """
        
        self.sdim = sdim
        self.udim = udim
        
        self.observation_space = spaces.box.Box(np.zeros(sdim), np.ones(sdim))
        self.action_space = spaces.box.Box(np.zeros(udim), np.ones(udim))
        
        self.goal = np.random.random(size=self.udim) if goal is None else goal
    
    def __repr__(self):
        
        return "TargetPractice(sdim=%s, udim=%s, goal=%r)" % (self.sdim, self.udim, self.goal)
        
    def reset(self):
        
        self.state = np.random.random(size=self.sdim)
        
        return self.state
    
    def dynamics(self, state, action):
        
        return np.random.random(size=self.sdim)
    
    def step(self, action):
        
        # the reward depends on the *old* action, not the new one:
        distance = np.sum((action -  self.goal) ** 2)
        reward = -distance

        # after having thus computed the reward, select the next state:
        self.state = self.dynamics(self.state, action)

        return self.state, reward, False, {} # obs, reward, done, oddities
    
    def render(self):
        
        raise NotImplementedError

    def close(self):
        
        raise NotImplementedError


class EchoGame(object):
    
    def __init__(self, sdim=3, udim=3):
        """ A game in which the goal is to repeat back the state as an action. """
        
        assert sdim == udim
        
        self.dim = self.sdim = self.udim = sdim
        
        self.observation_space = spaces.box.Box(np.zeros(self.dim), np.ones(self.dim))
        self.action_space = spaces.box.Box(np.zeros(self.dim), np.ones(self.dim))
    
    def __repr__(self):
        
        return "EchoGame(sdim=%s, udim=%s)" % (self.sdim, self.udim)
        
    def reset(self):
        
        self.state = np.random.random(size=self.sdim)
        
        return self.state
    
    def dynamics(self, state, action):
        
        return np.random.random(size=self.sdim)
    
    def step(self, action):
        
        # the reward depends on the *old* action, not the new one:
        distance = np.sum((self.state - action) ** 2)
        reward = -distance

        # after having thus computed the reward, select the next state:
        self.state = self.dynamics(self.state, action)

        return self.state, reward, False, {} # obs, reward, done, oddities
    
    def render(self):
        
        raise NotImplementedError

    def close(self):
        
        raise NotImplementedError


