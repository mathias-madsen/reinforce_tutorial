import numpy as np
import os
import time

from regressors import PolynomialRegressor
import logger


class ReinforceAgent(object):
    
    def __init__(self, policy):
        
        self.policy = policy
        self.baseline = PolynomialRegressor(self.policy.sdim)

        self.rsumlists = [] # progress tracker; saves N rewardsums per epoch
        self.purge()
    
    def purge(self):
        """ Clean out the short-term memory of recent rollouts. """
        
        self.stateslists = []
        self.actionslists = []
        self.rewardslists = []
        self.dlogproblists = []
        
        self.allstates = []
        self.alladvantages = []
    
    def rollout(self, environment, T=100, render=False, fps=24):
        """ Peform a single rollout in the given environment. """
        
        environment.reset()
        prev_action = environment.action_space.sample()

        states = []
        actions = [prev_action] # the action at time t - 1
        rewards = []
        scores = []

        for t in range(T):

            state, reward, done, info = environment.step(prev_action)
            action = self.policy.sample(state, prev_action)
            score = self.policy.dlogprob(action, self.policy.theta, state, prev_action)
            
            if render:
                environment.render()
                time.sleep(1.0/fps)
    
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            scores.append(score)
            
            prev_action = action
            
            if done:
                break
        
        return states, actions, rewards, scores
    
    def reinforce(self, states, rewards, scores, gamma=None):
        """ Compute (the REINFORCE gradient estimate, the array of advantages). """
        
        returns = self.smear(rewards, gamma=gamma)
        advantages = returns - self.baseline.predict(states)
        terms = (np.array(scores).T * advantages).T
        
        return np.sum(terms, axis=0), advantages
    
    def collect(self, environment, N=20, T=100, gamma=None, verbose=True):
        """ Collect learning-relevant stats over N rollouts of length <= T. """
        
        gradients = []
        rewardsums = []
        
        allstates = []
        alladvantages = []

        for n in range(N):
            
            states, actions, rewards, scores = self.rollout(environment, T=T)
            gradient, advantages = self.reinforce(states, rewards, scores, gamma)

            gradients.append(gradient)
            rewardsums.append(np.sum(rewards))
            
            allstates.extend(states)
            alladvantages.extend(advantages)

        meangradient = np.mean(gradients, axis=0)
        
        return meangradient, rewardsums, allstates, alladvantages

    def train(self, environment, I=100, N=100, T=1000, gamma=0.90, learning_rate=0.001,
                    verbose=True, dirpath=None, save_args=True, save_theta=True,
                    plot_progress=True, imshow_theta=True):
        """ Collect empirical information and update parameters I times. """
        
        if save_args or save_theta or plot_progress or imshow_theta:
            dirpath = logger.makedir(dirpath)
        
        if save_args:
            logger.save_args(dirpath, policy=self.policy, baseline=self.baseline,
                             environment=environment, I=I, N=N, T=T, gamma=gamma,
                             learning_rate=learning_rate)
        
        if plot_progress:
            rsumlists = []
        
        for i in range(I):
            
            if verbose:
                print("Training epoch i=%s:\n" % i)

            # obtain learning-relevant statistics through experimentation:
            gradient, rsums, states, advans = self.collect(environment, N, T, gamma)

            if verbose:
                logger.print_stats(rsums)

            # update the policy parameters as suggested by the gradient:
            self.policy.update(gradient, learning_rate, verbose=verbose)

            # Re-estimate the parameters of the advantage-predictor:
            self.baseline.fit(states, advans, verbose=verbose)

            # Having thus learned from the memory contents, discard it:
            self.purge()

            if save_theta:
                filename = os.path.join(dirpath, "theta.npz")
                self.policy.saveas(filename)

            if imshow_theta:
                filename = os.path.join(dirpath, "theta_%s.png" % str(i).rjust(4, '0'))
                self.policy.imshow_theta(show=False, filename=filename)

            if plot_progress:
                rsumlists.append(rsums)
                filename = os.path.join(dirpath, "progress.png")
                logger.plot_progress(rsumlists, show=False, filename=filename)

        if verbose:
            print(" Finished training. ".center(72, "="), "\n")

    def smear(self, rewards, gamma=None):
        """ Form returns from the rewards. """
        
        # In the plan REINFORCE algorithm, every action in the rollout is
        # held responsible for everything that happened at any time during
        # the rollout, whether past, present, or futre. We express this by
        # multiplying each score by the total sum-of-rewards.
        
        if gamma is None:
            return np.sum(rewards) * np.ones_like(rewards)

        # In order to reduce variance, however, it is safe to not hold any
        # action accountable for a past event (gamma == 1). If the environ-
        # ment has few long-term dependencies, it can also be advisable to
        # not hold any actions responsible for much later events (gamma < 1).
        
        returns = np.zeros_like(rewards)
        all_later_returns = 0.0
        
        for t, reward in reversed(list(enumerate(rewards))):
            returns[t] = reward + (gamma * all_later_returns)
            all_later_returns = returns[t]
        
        return returns
