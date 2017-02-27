import numpy as np
from matplotlib import pyplot as plt

import gym

from ReinforceAgent import ReinforceAgent
from policies import PolynomialPolicy, FeedForwardPolicy
from Arm import ReachingGame


# game = gym.make('Pendulum-v0')
game = ReachingGame()

policy = FeedForwardPolicy(environment=game, hidden=[25, 25, 25])
agent = ReinforceAgent(policy=policy)

# agent.rollout(game, T=500, render=True)

agent.train(game, I=200, N=50, T=500, gamma=0.95, learning_rate=0.001, imshow_weights=False)

