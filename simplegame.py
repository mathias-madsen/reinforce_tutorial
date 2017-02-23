import numpy as np
from matplotlib import pyplot as plt

import gym

from ReinforceAgent import ReinforceAgent
from PolynomialPolicy import StickyDiscretePolicy
from environments import SimpleEnvironment


game = SimpleEnvironment()

sdim = game.observation_space.shape[0]
usize = game.action_space.n

policy = StickyDiscretePolicy(sdim=sdim, usize=usize, degree=0,
                              shuffleprob=0.0, repeatprob=0.0)

agent = ReinforceAgent(policy=policy)
agent.train(game, I=50, N=20, T=100, gamma=0.98)

