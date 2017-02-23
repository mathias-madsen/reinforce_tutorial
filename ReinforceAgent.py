import numpy as np

from matplotlib import pyplot as plt

from regressors import PolynomialRegressor

import os # for creating the results directory
import datetime # for naming the results directory
import time # for slowing down animations


class ReinforceAgent(object):
    
    def __init__(self, policy):
        
        self.policy = policy
        self.baseline = PolynomialRegressor(self.policy.sdim)

        self.rewardsums = [] # progress tracker
        self.purge()
    
    def purge(self):
        """ Clean out the short-term memory of recent rollouts. """
        
        self.stateslists = []
        self.actionslists = []
        self.rewardslists = []
        self.dlogproblists = []
    
    def rollout(self, environment, episodesteps=2000, render=False, fps=24):
        """ Peform a single rollout in the given environment. """
        
        environment.reset()
        prev_action = environment.action_space.sample()

        states = []
        actions = [prev_action] # the action at time t - 1
        rewards = []
        dlogprobs = []

        for t in range(episodesteps):

            state, reward, done, info = environment.step(prev_action)
            this_action = self.policy.sample(state, prev_action)
            
            dlogprob = self.policy.dlogprob(this_action, self.policy.theta, state, prev_action)
            
            if render:
                environment.render()
                time.sleep(1.0/fps)
    
            states.append(state)
            actions.append(this_action)
            rewards.append(reward)
            dlogprobs.append(dlogprob)
            
            prev_action = this_action
            
            if done:
                break
        
        return states, actions, rewards, dlogprobs

    def interact(self, environment, episodes=20, episodesteps=2000, verbose=True):
        """ Peform multiple rollouts in the given environment. """
        
        if verbose:
            print("Performing %s rollouts . . ." % episodes, end=" ", flush=True)
            
        for episode in range(episodes):
            
            states, actions, rewards, dlogprobs = self.rollout(environment, episodesteps=episodesteps)
            
            self.stateslists.append(states)
            self.actionslists.append(actions)
            self.rewardslists.append(rewards)
            self.dlogproblists.append(dlogprobs)
            
        sums = [np.sum(rewards) for rewards in self.rewardslists]
        self.rewardsums.append(sums)

        if verbose:
            print("Done.\n")
            self.printstats(sums)
    
        return self.stateslists, self.actionslists, self.rewardslists, self.dlogproblists

    def printstats(self, rewardsums):
        """ Pretty-print some statistical information about the data. """

        print("Percentiles of the episode-wise sums-of-rewards:\n")
    
        percents = np.linspace(0, 100, 5 + 1)
        percentiles = [np.percentile(rewardsums, p) for p in percents]
    
        print(" | ".join(("%.0f" % p).center(9) for p in percents))
        print(" | ".join(("%.5g" % p).center(9) for p in percentiles))
        print()
        
        meantext = "Mean sum-of-rewards: %.3f." % np.mean(rewardsums)

        print(meantext)
        print("‾" * len(meantext))
        print()
        
    def plot_progress(self, show=True, filename=None):
        """ Show how the distribution of reward sums has developed. """
        
        plt.figure(figsize=(20, 10))
        
        numepochs = len(self.rewardsums)
        epochs = np.arange(numepochs)
        
        means = [np.mean(sumlist) for sumlist in self.rewardsums]
        medians = [np.median(sumlist) for sumlist in self.rewardsums]

        for p in np.linspace(5, 50, 10):
            
            top = [np.percentile(sumlist, 50 + p) for sumlist in self.rewardsums]
            bot = [np.percentile(sumlist, 50 - p) for sumlist in self.rewardsums]
            
            plt.fill_between(epochs, bot, top, color="gold", alpha=0.1)
        
        plt.plot(epochs, medians, color="orange", alpha=0.5, lw=5)
        plt.plot(epochs, means, color="blue", alpha=0.5, lw=4)
        
        plt.xlim(-1, numepochs)
        
        plt.xlabel("Training epoch", fontsize=24)
        plt.ylabel("Sum-of-rewards per episode", fontsize=24)
        
        if filename:
            plt.savefig(filename)
        
        if show:
            plt.show()
    
        plt.close('all')
    
    def smear(self, rewards, gamma=1.0):
        """ Form returns from the rewards. """

        returns = np.zeros_like(rewards)
        future_return = 0.0
        
        for t, reward in reversed(list(enumerate(rewards))):
            returns[t] = reward + gamma*future_return
            future_return = returns[t]
        
        return returns
        
    def advantages(self, stateslists, rewardslists, gamma=1.0):
        """ Compute the baselined returns from the raw rewards. """

        returns = [self.smear(rewards, gamma=gamma) for rewards in rewardslists]
        expectations = [self.baseline.predict(states) for states in stateslists]

        return [obs - mean for obs, mean in zip(returns, expectations)]

    def learn(self, learning_rate=1.0, gamma=1.0, verbose=True):
        
        if verbose:
            print("LEARNING:")
            print("‾‾‾‾‾‾‾‾‾")
            print("Computing baselined returns from rewards . . .", end=" ", flush=True)

        advantages = self.advantages(self.stateslists, self.rewardslists, gamma=gamma)

        if verbose:
            print("Done.")
            print("Estimating gradient . . .", end=" ", flush=True)

        surlists = self.stateslists, self.actionslists, advantages

        unfiltered = [self.policy.REINFORCE(*sur) for sur in zip(*surlists)]
        filtered = [vector for vector in unfiltered if not np.any(np.isnan(vector))]
        
        if filtered is not []:
            gradient = np.mean(filtered, axis=0)
        else:
            gradient = np.zeros_like(self.policy.theta)
            print("¡¡¡ WARNING: contains nans !!!", end=" ", flush=True)
        
        assert not np.any([np.isnan(vector) for vector in filtered])
        assert not np.any(np.isnan(gradient))
        
        if verbose:
            length = np.sum(gradient ** 2) ** 0.5
            print("Gradient length: %.5g.\n" % length)

        assert self.policy.theta.shape == gradient.shape
        
        old_theta = np.copy(self.policy.theta)
        
        step = learning_rate * gradient
        self.policy.theta = self.policy.theta + step
        
        new_theta = np.copy(self.policy.theta)
        
        print("Old theta:\n", old_theta, "\n")
        print("New theta:\n", new_theta, "\n")

        if verbose:
            L2 = np.sum(step ** 2) ** 0.5
            incr = np.sum(step * gradient)
            print("Took a step of length %.5g Euclidean units." % L2)
            print("Expecting an improvement of %.5g return units." % incr)
            print()
        
        allstates = np.concatenate(self.stateslists)
        alladvantages = np.concatenate(advantages)

        self.baseline.fit(allstates, alladvantages, verbose=verbose)
        self.purge() # this was the last thing we needed the data for
        
        if verbose:
            print("Done. The memory of the agent has now been purged.\n")
    
    def train(self, environment, epochs=100, episodes=100, episodesteps=1000,
                    learning_rate=0.001, gamma=0.90, directory=None):
        """ Repeatedly update the policy over several training epochs. """
        
        # if no path is given, make a subdirectory where the script is called
        if directory is None:
            now = datetime.datetime.now()
            directory = now.strftime("results/%Y_%b_%d_%Hh%M")

        # if not exists, create
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # log parameters
        calllog = open("%s/call.txt" % directory, "w")
        calllog.write("""Call parameters:
        policy = %s
        baseline = %s
        environment = %s
        epochs = %s
        episodes = %s
        episodesteps = %s
        learning_rate = %s
        gamma = %s
        """ % (self.policy, self.baseline, environment, epochs,
               episodes, episodesteps, learning_rate, gamma))
        calllog.close()
        
        for epoch in range(epochs):

            print((" TRAINING EPOCH %s " % (epoch + 1)).center(72, '='), "\n")
            
            ######## CENTRAL DATA COLLECTION AND LEARNING CALLS ########

            # in order to distinguish flukes from progress, collect twice:
            self.interact(environment, episodes=episodes, episodesteps=episodesteps // 2)
            self.interact(environment, episodes=episodes, episodesteps=(episodesteps + 1) // 2)

            # then learn from all that you've seen:
            self.learn(learning_rate=learning_rate, gamma=gamma)
            
            ############################################################

            # save one more picture of the parameter matrix
            numstr = str(epoch).rjust(4, '0')
            imshowname = "%s/theta_%s.png" % (directory, numstr)
            self.policy.imshow_theta(show=False, filename=imshowname)
            
            # overwrite the existing progress plot
            plotname = "%s/progress.png" % directory
            self.plot_progress(show=False, filename=plotname)
            
            # overwrite the existing policy pickle
            npzname = "%s/trained_policy.npz" % directory
            self.policy.saveas(npzname)
        
        print(" Finished training. ".center(72, "="))
        print()
