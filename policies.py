import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as tns

import distributions


class Policy(object):
    
    def __init__(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def __repr__(self):
        
        classname = self.__class__.__name__
        keyvalues = ["%s=%s" % item for item in self.__dict__.items()]
        
        return "%s(%s)" % (classname, "{%s}" % ", ".join(keyvalues))

    def saveas(self, filename=None):
        """ Save the parameters of the policy for later loading. """
        
        np.savez(filename, **self.__dict__)
    
    def load(self, filename=None):
        """ Initialize a policy from saved parameters. """
        
        self.__dict__.update(np.load(filename))

    def imshow_weights(self, show=True, filename=None):
        """ imshow the parameter matrix. """

        plt.imshow(self.weights, aspect='auto', interpolation='nearest')
        plt.colorbar()
        
        if filename is not None:
            plt.savefig(filename)
        
        if show:
            plt.show()
        
        plt.close('all')
    
    def update(self, direction, length, verbose=False):
        """ Take a step in the direction, peforming certain sanity checks. """
        
        assert not np.any(np.isnan(direction))
        assert direction.shape == self.weights.shape
        
        if verbose:
            L2 = np.sum(direction ** 2) ** 0.5
            print("Length of direction vector: %.5g." % L2)
            print("Length of the step taken: %.5g." % (length * L2))
            print()

        self.weights += length * direction


class ContinuousPolicy(Policy):
    
    def __init__(self, sdim=None, udim=None, low=None, high=None, degree=3,
                       weights=None, filename=None):

        if udim is None: # we allow the udim to be inferred from low and high

            udim = len(low) if low is not None else None
            udim = len(high) if high is not None else None
        
        if udim is not None: # can still fail if low == high == udim == None

            low = np.zeros(udim) if low is None else low
            high = np.ones(udim) if high is None else high
        
        if sdim is not None and degree is not None and weights is None:
            
            inputdim = 1 + (sdim * degree) # [1.0] + concatenated powers
            outputdim = 2 # a Beta distribution has two parameters

            weights = np.zeros((outputdim, udim, inputdim))
            weights[:, :, 0] = 1
        
        self.sdim = sdim
        self.udim = udim

        self.low = low
        self.high = high

        self.degree = degree
        self.weights = weights

        if filename is not None:
            self.load(filename)
        
        assert np.all(self.low <= self.high)

        self.compile()
    
    def DESIGN(self, SHIST, UHIST):
        
        STATE = SHIST[-1, :]

        return tns.concatenate([[1]] + [STATE ** (n + 1) for n in range(self.degree)])
    
    def PARAMS(self, SHIST, UHIST, WEIGHTS):
        
        INPUTS = self.DESIGN(SHIST, UHIST)
        PRODUCT = tns.dot(WEIGHTS, INPUTS)
        
        return tns.abs_(PRODUCT)
    
    def LOGP(self, ACTION, PARAMS):
        
        return self.dist.LOGP(ACTION, PARAMS)
    
    def compile(self):
        
        SHIST = tns.dmatrix("STATE HISTORY")
        UHIST = tns.dmatrix("ACTION HISTORY")
        WEIGHTS = tns.dtensor3("WEIGHTS")
        
        PARAMS = self.PARAMS(SHIST, UHIST, WEIGHTS)

        ACTION = tns.dvector("ACTION")
        
        SQUASHED = self.squash(ACTION)
        LOGP = self.LOGP(SQUASHED, PARAMS)
        SCORE = theano.grad(LOGP, wrt=WEIGHTS)
        
        print("Compiling policy functions . . .")
        self.design = theano.function(inputs=[SHIST, UHIST], outputs=self.DESIGN(SHIST, UHIST), on_unused_input='ignore')
        self.params = theano.function(inputs=[SHIST, UHIST, WEIGHTS], outputs=PARAMS, on_unused_input='ignore')
        self.dlogp = theano.function(inputs=[ACTION, SHIST, UHIST, WEIGHTS], outputs=SCORE, on_unused_input='ignore')
        print("Done.\n")
        
    def squash(self, normalized_action):
        """ Stretch and move a unit cube action to fit into the action box. """

        raise NotImplementedError

    def unsquash(self, action):
        """ Normalize an action so that becomes a point in the unit cube. """

        raise NotImplementedError

    def sample(self, shist, uhist, weights=None):
        
        if weights is None:
            weights = self.weights
        
        shist = np.atleast_2d(shist)
        uhist = np.atleast_2d(uhist)
        
        params = self.params(shist, uhist, weights)
        normalized_action = self.dist.sample(params)
        
        return self.squash(normalized_action)

    def score(self, action, shist=[], uhist=[], weights=None):
        
        if weights is None:
            weights = self.weights
        
        shist = np.atleast_2d(shist)
        uhist = np.atleast_2d(uhist)
        
        return self.dlogp(action, shist, uhist, weights)


class BetaPolicy(ContinuousPolicy):
    
    def __init__(self, *args, **kwargs):
        
        self.dist = distributions.Beta()

        super().__init__(*args, **kwargs)

    def squash(self, normalized_action):
        """ Stretch and move a unit cube action to fit into the action box. """
        
        return self.low + (self.high - self.low)*normalized_action
    
    def unsquash(self, action):
        """ Normalize an action so that becomes a point in the unit cube. """
        
        return (action - self.low) / (self.high - self.low)


class GaussianPolicy(ContinuousPolicy):
    
    def __init__(self, *args, **kwargs):
        
        self.dist = distributions.Gaussian()

        super().__init__(*args, **kwargs)

    def squash(self, sample_vector):
        """ Squash a vector of reals into the action box. """
        
        normalized_action = 0.5*(1 + np.arctan(sample_vector))
        
        return self.low + (self.high - self.low)*normalized_action
    
    def unsquash(self, action):
        """ Unsquash an element from the action box to a vector of reals. """
        
        normalized_action = (action - self.low) / (self.high - self.low)
        
        return 2*np.tan(normalized_action) - 1


if __name__ == '__main__':
    
    import gym
    
    env = gym.make('Pendulum-v0')
    
    policy = BetaPolicy(sdim=3, udim=5)
    
    s = 2 + np.arange(policy.sdim)
    
    print("s.shape:")
    print(s.shape)
    print()    
    print("design(s).shape:")
    print(policy.design([s], [[]]).shape)
    print()
    print("policy.weights.shape:")
    print(policy.weights.shape)
    print()
    print("Params(s, weights):")
    print(policy.params([s], [[]], policy.weights))
    print()
    
    u = policy.sample([s], [[]])
    
    print("Sample:")
    print(u)
    print()
    print("dlogp(u):")
    print(policy.score(u, [s], [[]], policy.weights))
    print()