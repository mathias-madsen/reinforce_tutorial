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

# # the winning policy:
# policy.theta = np.zeros((usize, sdim + 1))
# policy.theta[0, 0] = +1.0
# policy.theta[1, 0] = -1.0

agent = ReinforceAgent(policy=policy)


steps = 100
iterations = 5

print("The episode length is %s, and we collect %s episodes." % (steps, iterations))
print()

for iteration in range(iterations):
    
    print((" ITERATION %s " % iteration).center(60, '='))
    print()

    estimates = []
    rewardsums = []

    print("Current theta setting:")
    print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    print(np.around(policy.theta, 4))
    print()

    for i in range(100):

        states, actions, rewards, dlogps = agent.rollout(game, episodesteps=steps)
    
        rewardsum = np.sum(rewards)
        estimate = np.sum(rewardsum * np.array(dlogps), axis=0)
    
        estimates.append(estimate)
        rewardsums.append(rewardsum)

    mean, std = np.mean(rewardsums), np.std(rewardsums)
    rewardstats = mean, std, mean / steps, (std**2 / steps)**0.5

    gradient = np.mean(estimates, axis=0)
    variance = np.mean([np.sum((v - gradient)**2) for v in estimates])

    print("Batch performance:")
    print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    print("%.3f ± %.3f (or %.3f ± %.3f per timestep)." % rewardstats)
    print()
    print("Mean gradient estimate:" % variance)
    print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    print(np.around(gradient, 4))
    print()
    print("(Variance = %.4g, or %.4g per timestep)." % (variance, variance/steps))
    print()
    
    policy.theta += 0.1*gradient