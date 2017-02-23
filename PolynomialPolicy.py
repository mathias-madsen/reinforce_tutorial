import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as tns


class StickyDiscretePolicy(object):
    
    def __init__(self, sdim=None, usize=None, repeatprob=0.9, shuffleprob=0.1,
                       degree=3, verbose=True, filename=None):
        
        if filename is None:
            
            self.repeatprob = repeatprob
            self.shuffleprob = shuffleprob
            self.degree = degree

            self.sdim = sdim
            self.usize = usize

            thetashape = self.usize, (self.degree + 1)*self.sdim + 1
            self.theta = np.random.normal(size=thetashape)

        else:
            
            self.load(filename)
        
        self.compile(verbose=verbose)
    
    def __repr__(self):
        
        classname = self.__class__.__name__
        description = classname, self.sdim, self.usize, self.repeatprob, self.degree
        
        return "<%s: sdim=%s, usize=%s, repeatprob=%s, degree=%s>" % description
    
    def saveas(self, filename=None):
        """ Save the parameters of the policy for later loading. """
        
        parameters = {
            'sdim': self.sdim,
            'usize': self.usize,
            'repeatprob': self.repeatprob,
            'degree': self.degree,
            'theta': self.theta
        }

        np.savez(filename, **parameters)
    
    def load(self, filename=None):
        """ Initialize a policy from saved parameters. """

        source = np.load(filename)
        
        self.sdim = source['sdim']
        self.usize = source['usize']
        self.repeatprob = source['repeatprob']
        self.degree = source['degree']
        self.theta = source['theta']

    def imshow_theta(self, show=True, filename=None):
        """ imshow the parameter matrix. """

        plt.imshow(self.theta, aspect='auto', interpolation='nearest')
        plt.colorbar()
        
        if filename is not None:
            plt.savefig(filename)
        
        if show:
            plt.show()
        
        plt.close('all')
    
    def compile(self, verbose=True):

        THETA = tns.dmatrix("theta")

        STATE = tns.dvector("S[t]") # the state that was just observed
        U_OLD = tns.iscalar("U[t - 1]") # previous action still remembered
        U_NEW = tns.iscalar("U[t]") # an observed action
        
        POWERS = [STATE ** n for n in range(self.degree + 1)]
        INPUT = tns.concatenate(POWERS + [[U_OLD]])
        
        # the possibility of selecting a new action based on the state:
        WEIGHTS = THETA.dot(INPUT)
        LOG_UNNORMALIZED = WEIGHTS - tns.max(WEIGHTS)
        UNNORMALIZED = tns.exp(LOG_UNNORMALIZED)
        NEW_P = UNNORMALIZED / UNNORMALIZED.sum()
        
        # the possibility of deterministically repeating the last action:
        REPEAT = tns.zeros(self.usize)
        REPEAT = tns.set_subtensor(REPEAT[U_OLD], 1.0)

        # the possibility of blondly selecting a random state:
        UNIFORM = tns.ones(self.usize) / self.usize
        
        UNSHUFFLED = self.repeatprob*REPEAT + (1 - self.repeatprob)*NEW_P
        PROBS = self.shuffleprob*UNIFORM + (1 - self.shuffleprob)*UNSHUFFLED

        PROB = PROBS[U_NEW]
        LOGPROB = tns.log(PROB)
        DLOGPROB = theano.grad(LOGPROB, wrt=THETA)
        
        pastonly = [THETA, STATE, U_OLD]
        full = [U_NEW, THETA, STATE, U_OLD]
        
        if verbose:
            print("Compiling policy evaluation functions . . .")

        self.weights = theano.function(inputs=pastonly, outputs=WEIGHTS)
        self.probs = theano.function(inputs=pastonly, outputs=PROBS)
        self.prob = theano.function(inputs=full, outputs=PROB)
        self.logprob = theano.function(inputs=full, outputs=LOGPROB)
        self.dlogprob = theano.function(inputs=full, outputs=DLOGPROB)
        
        if verbose:
            print("Done.\n")
    
    def distribution(self, s, u):
        """ Probability vector for u[t] given s[t], u[t - 1], and theta. """
        
        probabilities = self.probs(self.theta, s, u) # theta for free
        
        assert np.all(probabilities >= 0)
        assert np.all(probabilities - 1 <= 1e-7) # almost smaller than 1
        assert np.abs(1.0 - np.sum(probabilities)) < 1e-7 # almost sum to 1
        assert not np.any(np.isnan(probabilities)) # are not nan
        assert np.allclose([np.sum(probabilities)], [1.0]) # almost sum to 1

        return probabilities / probabilities.sum() # note the test above
    
    def sample(self, s, u):
        """ Sample u[t] given s[t], u[t - 1], and theta. """
        
        return np.random.choice(range(self.usize), p=self.distribution(s, u))

    def update(self, direction, length, verbose=False):
        """ Take a step in the direction, peforming certain sanity checks. """
        
        assert not np.any(np.isnan(direction))
        assert direction.shape == self.theta.shape
        
        if verbose:
            L2 = np.sum(direction ** 2) ** 0.5
            print("Length of direction vector: %.5g." % L2)
            print("Length of the step taken: %.5g." % (length * L2))
            print()

        self.theta += length * direction


if __name__ == '__main__':
    
    sdim = 2
    usize = 3
    degree = 1
    
    repeatprob = 0.01
    shuffleprob = 0.01
    
    policy  = StickyDiscretePolicy(sdim=sdim, usize=usize, degree=degree,
                                   repeatprob=repeatprob, shuffleprob=shuffleprob)

    udim = 1 # by definition
    powerdim = sdim * (degree + 1) # +1 by linguistic convention
    inputdim = powerdim + udim
    
    assert policy.theta.shape == (usize, inputdim)

    policy.theta[:, 0:2] = 0 # constant terms
    policy.theta[:, 2:4] = np.arange(6).reshape((usize, sdim)) # linear terms
    policy.theta[:, 4:5] = 0 # repeat-last terms
    
    print("theta:\n\n", policy.theta, "\n")
    
    for svalues in [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]:
    
        u0 = 0
        s1 = np.array(svalues)
        u1 = policy.sample(s1, u0)

        weights = policy.weights(policy.theta, s1, u0)
        probs = policy.probs(policy.theta, s1, u0)
        dlogprob = policy.dlogprob(u1, policy.theta, s1, u0)
        
        skewed = np.exp(weights) / np.exp(weights).sum()

        deterministic = np.zeros(usize)
        deterministic[u0] = 1
        
        mixedprobs = repeatprob*deterministic + (1 - repeatprob)*skewed
        
        uniform = np.ones(usize) / usize
        
        numprobs = shuffleprob*uniform + (1 - shuffleprob)*mixedprobs

        print(probs)
        print(numprobs)
        print()

        assert np.allclose(probs, numprobs)

        print("Unnormalized weights, w1:", weights)
        print("Probability vector, p1:", np.around(probs, 2))
        print("Selected action, u1:", u1)
        print()
        print("D_theta logprob(u1=%s | s1=%s, u0=%s, theta):\n" % (u1, s1, u0))
        print(dlogprob, "\n")
        print()
    
    stepsize = 1.0

    old_theta = policy.theta
    old_logp = policy.logprob(u1, old_theta, s1, u0)

    gradient = policy.dlogprob(u1, old_theta, s1, u0)
    step = stepsize*gradient

    new_theta = policy.theta = old_theta + step
    new_logp = policy.logprob(u1, new_theta, s1, u0)
    
    print("Increase in logp (epsilon=%.3f):" % stepsize)
    print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
    print("Expected:", np.sum(step * gradient))
    print("Observed:", new_logp - old_logp)
    print()
