import numpy as np
import os
import time

import logger
from BlockyVector import BlockyVector
from regressors import PolynomialRegressor, PolynomialTemporalRegressor


class ReinforceAgent(object):
    
    def __init__(self, policy, baseline=None):
        
        self.policy = policy
        
        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = PolynomialRegressor(sdim=self.policy.sdim, degree=0)

        self.rsumlists = [] # progress tracker; saves N rewardsums per epoch

    def rollout(self, environment, T=100, render=False, fps=24):
        """ Peform a single rollout in the given environment. """
        
        smemory = 5
        umemory = 5
        
        environment.reset() # to ensure the existence of a first state

        states = [environment.reset() for t in range(smemory)]
        actions = [environment.action_space.sample() for t in range(umemory)]
        rewards = []
        scores = []

        for t in range(T):

            if render:
                environment.render()
                time.sleep(1.0/fps)

            # The agent responds to the environment:
            action = self.policy.sample(states, actions, self.policy.weights)
            score = self.policy.score(action, states, actions, self.policy.weights)

            actions.append(action)
            scores.append(score)

            # The environment responds to the agent:
            state, reward, done, info = environment.step(action)

            states.append(state)
            rewards.append(reward)

            if done:
                break
        
        if render:
            environment.close()
        
        # Because of the state yielded from the initial environment.reset(),
        # we end up with one state which the agent never gets to respond to:
        states = states[smemory - 1 : T + smemory - 1]
        actions = actions[umemory:]
        
        assert len(states) == len(actions) == T

        return states, actions, rewards, scores
    
    def reinforce(self, states, rewards, scores, gamma=None):
        """ Compute (the REINFORCE gradient estimate, the array of advantages). """
        
        returns = self.smear(rewards, gamma=gamma)
        advantages = returns - self.baseline.predict(states)

        assert not np.any(np.isnan(returns))
        assert not np.any(np.isnan(advantages))
        
        # Note: the * in `adv * score` triggers score.__rmul__(adv).

        terms = [adv * score for adv, score in zip(advantages, scores)]
        gradient = BlockyVector(np.sum(terms, axis=0))
        
        assert gradient.shape == self.policy.weights.shape
        
        return gradient, advantages
    
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
        
        meangradient = BlockyVector(np.mean(gradients, axis=0))

        if verbose:
            
            length = (meangradient ** 2).sum() ** 0.5
            print("Length of the mean gradient: %.2f." % length)
            
            sqs = [((BlockyVector(g) - meangradient) ** 2).sum() for g in gradients]
            std = np.mean(sqs, axis=0) ** 0.5
            print("Standard deviation of the sample gradients: %.2f." % std)

            print()

        return meangradient, rewardsums, allstates, alladvantages

    def train(self, environment, I=np.inf, N=100, T=1000, gamma=0.90, learning_rate=0.1,
                    verbose=True, dirpath=None, save_args=True, save_weights=True,
                    plot_progress=True, imshow_weights=False):
        """ Collect empirical information and update parameters I times. """
        
        if save_args or save_weights or plot_progress or imshow_weights:
            dirpath = logger.makedir(dirpath)
        
        if save_args:
            logger.save_args(dirpath, policy=self.policy, baseline=self.baseline,
                             environment=environment, I=I, N=N, T=T, gamma=gamma,
                             learning_rate=learning_rate, dist=self.policy.dist)

        if plot_progress:
            rsumlists = []
        
        if imshow_weights:
            filename = os.path.join(dirpath, "weights_0000.png")
            self.policy.imshow_weights(show=False, filename=filename)

        if save_weights:
            new_named_file = os.path.join(dirpath, "weights_0000.npz")
            old_most_recent = os.path.join(dirpath, "weights.npz")
            self.policy.saveas(new_named_file)
            self.policy.saveas(old_most_recent)

        i = 0

        while True:
            
            if verbose:
                print("\n", ("TRAINING EPOCH %s:" % i).center(60), "\n")

            # obtain learning-relevant statistics through experimentation:
            gradient, rsums, states, advans = self.collect(environment, N, T, gamma)

            if verbose:
                logger.print_stats(rsums)

            # update the policy parameters as suggested by the gradient:
            self.policy.update(gradient, learning_rate, verbose=verbose)

            # Re-estimate the parameters of the advantage-predictor:
            self.baseline.fit(states, advans, verbose=verbose)

            numi = str(i + 1).rjust(4, '0')
            
            if save_weights:
                new_named_file = os.path.join(dirpath, "weights_%s.npz" % numi)
                old_most_recent = os.path.join(dirpath, "weights.npz")
                self.policy.saveas(new_named_file)
                self.policy.saveas(old_most_recent)

            if imshow_weights:
                filename = os.path.join(dirpath, "weights_%s.png" % numi)
                self.policy.imshow_weights(show=False, filename=filename)

            if plot_progress:
                rsumlists.append(rsums)
                filename = os.path.join(dirpath, "progress.png")
                logger.plot_progress(rsumlists, show=False, filename=filename)
            
            i += 1
            if i >= I:
                break

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


if __name__ == '__main__':
    
    import policies
    
    agent = ReinforceAgent(policies.PolynomialPolicy(sdim=2, udim=3, degree=0))