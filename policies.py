import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as tns

import distributions
from BlockyVector import BlockyVector


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
        
        source = np.load(filename)
        methods = 'dist design dlogp params'.split(' ')
        
        self.__dict__.update({name: entry for name, entry in source.items()
                              if name not in methods})

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

        assert direction.shape == self.weights.shape
        
        if verbose:
            L2 = (direction ** 2).sum() ** 0.5
            print("Length of direction vector: %.5g." % L2)
            print("Length of the step taken: %.5g." % (length * L2))
            print()

        self.weights += length * direction


class ContinuousPolicy(Policy):
    
    def __init__(self, environment=None, weights=None, Dist=distributions.Gaussian,
                       filename=None, *args, **kwargs):
        
        if filename is not None:
            self.load(filename)
            self.compile()
            return
        
        self.sdim, = environment.observation_space.shape
        self.udim, = environment.action_space.shape
        
        self.low = environment.action_space.low
        self.high = environment.action_space.high

        self.weights = weights
        self.dist = Dist() # make new instance: Dist.__init__() == dist
        
        if weights is not None:
            self.weights = BlockyVector(weights)
        else:
            self.initialize_weights()

        self.compile()
    
    def wrap(self, action):
        """ Convert an unnormalized action into a box element. """

        return self.low + (self.high - self.low)*self.dist.squash(action)
        
    def WRAP(self, action):
        """ Convert an unnormalized action into a box element. """

        return self.low + (self.high - self.low)*self.dist.SQUASH(action)

    def unwrap(self, box_elm):
        """ Convert a box element into a unit cube element. """

        return self.dist.unsquash((box_elm - self.low) / (self.high - self.low))
    
    def UNWRAP(self, box_elm):
        """ Convert a box element into a unit cube element. """

        return self.dist.UNSQUASH((box_elm - self.low) / (self.high - self.low))

    def LOGP(self, ACTION, PARAMS):
        
        return self.dist.LOGP(ACTION, PARAMS)
    
    def compile(self):
        
        SHIST = tns.dmatrix("STATE HISTORY")
        UHIST = tns.dmatrix("ACTION HISTORY")
        
        WEIGHTS = [tns.dmatrix("WEIGHT_%s" % i) for i in range(len(self.weights))]
        PARAMS = self.PARAMS(SHIST, UHIST, WEIGHTS)

        ACTION = tns.dvector("ACTION")
        
        SAMPLE = self.UNWRAP(ACTION)
        LOGP = self.LOGP(SAMPLE, PARAMS)
        SCORE = theano.grad(LOGP, wrt=WEIGHTS)
        
        print("Compiling policy functions . . .")
        self.params = theano.function(inputs=[SHIST, UHIST] + WEIGHTS, outputs=PARAMS, on_unused_input='ignore')
        self.dlogp = theano.function(inputs=[ACTION, SHIST, UHIST] + WEIGHTS, outputs=SCORE, on_unused_input='ignore')
        print("Done.\n")
        
    def sample(self, shist, uhist, weights=None):
        
        if weights is None:
            weights = self.weights
        
        shist = np.atleast_2d(shist)
        uhist = np.atleast_2d(uhist)
        
        params = self.params(shist, uhist, *weights)
        unnormalized = self.dist.sample(params)
        
        return self.wrap(unnormalized)

    def score(self, action, shist=[], uhist=[], weights=None):
        
        if weights is None:
            weights = self.weights
        
        shist = np.atleast_2d(shist)
        uhist = np.atleast_2d(uhist)
        
        return BlockyVector(self.dlogp(action, shist, uhist, *weights))


class PolynomialPolicy(ContinuousPolicy):
    
    def __init__(self, degree=3, *args, **kwargs):
        
        self.degree = degree
        
        super().__init__(*args, **{key: val for key, val in kwargs.items() if key != 'degree'})
    
    def initialize_weights(self):

        inputdim = 1 + (self.sdim * self.degree) # [1] + concatenated powers
        outputdim = 2 # number of parameters of the action distribution
        
        muweights = np.random.normal(size=(self.udim, inputdim))
        sigmaweights = np.random.normal(size=(self.udim, inputdim))
        
        self.weights = BlockyVector([muweights, sigmaweights])

    def PARAMS(self, SHIST, UHIST, WEIGHTS):
        
        STATE = SHIST[-1, :]
        
        POWERS = [STATE ** (n + 1) for n in range(self.degree)]
        INPUTS = tns.concatenate([[1]] + POWERS)

        return [tns.dot(WEIGHT, INPUTS) for WEIGHT in WEIGHTS]


class FeedForwardPolicy(ContinuousPolicy):
    
    def __init__(self, hidden=[], *args, **kwargs):
        
        self.hidden = hidden
        
        super().__init__(*args, **{key: val for key, val in kwargs.items() if key != 'hidden'})
    
    def initialize_weights(self):
        
        indims = [self.sdim] + self.hidden
        outdims = self.hidden + [self.udim]
        
        weights = [np.random.normal(size=(outdim, indim + 1))
                   for indim, outdim in zip(indims, outdims)]
        
        self.weights = BlockyVector(weights)

    def PARAMS(self, SHIST, UHIST, WEIGHTS):
        
        INPUT = SHIST[-1, :]

        X = [INPUT]
        
        for WEIGHT in WEIGHTS[:-1]:
            LAYER = tns.concatenate([X[-1], [1]])
            COMBINATION = tns.dot(WEIGHT, LAYER)
            X.append(tns.tanh(COMBINATION))

        LAYER = tns.concatenate([X[-1], [1]])
        COMBINATION = tns.dot(WEIGHTS[-1], LAYER)

        return COMBINATION

#
# class PolynomialGaussianPolicy(PolynomialPolicy):
#
#     def __init__(self, *args, **kwargs):
#
#         self.dist = distributions.Gaussian()
#
#         super().__init__(*args, **kwargs)


if __name__ == '__main__':
    
    import gym
    
    env = gym.make('Pendulum-v0')
    
    policy = PolynomialBetaPolicy(sdim=3, udim=5)
    
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