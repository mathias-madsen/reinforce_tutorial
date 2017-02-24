import numpy as np
from matplotlib import pyplot as plt

import gym

from ReinforceAgent import ReinforceAgent
from policies import ContinuousPolicy, BetaPolicy, GaussianPolicy
from environments import SimpleEnvironment


game = gym.make('Pendulum-v0')

sdim = game.observation_space.shape[0]
udim = game.action_space.shape[0]

low = game.action_space.low
high = game.action_space.high

policy = GaussianPolicy(sdim=sdim, udim=udim, low=low, high=high, degree=2)
agent = ReinforceAgent(policy=policy)

agent.train(game, I=100, N=100, T=1000, gamma=0.90, learning_rate=1e-6, imshow_weights=False)

