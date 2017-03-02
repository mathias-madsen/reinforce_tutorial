import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as tns

import distributions
from BlockyVector import BlockyVector


class Policy(object):
    
    def __init__(self, *args, **kwargs):
        
        pass
    
    def __repr__(self):
        
        classname = self.__class__.__name__
        keyvalues = ["%s=%s" % item for item in self.__dict__.items()]
        
        return "%s(%s)" % (classname, "{%s}" % ", ".join(keyvalues))

    def saveas(self, filename=None):
        """ Save the parameters of the policy for later loading. """
        
        np.savez(filename, **self.__dict__)
    
    def load(self, filename=None):
        """ Initialize a policy from saved parameters. """
        
        for key, value in np.load(filename).items():
            self.__dict__[key] = value if value.ndim > 0 else value.item()

    def imshow_weights(self, blocks=None, show=True, filename=None):
        """ imshow the parameter matrix. """
        
        if blocks is None:
            blocks = self.weights

        nblocks = len(blocks)
        width, height = 16, 9

        assert nblocks > 0

        ncols = int(np.ceil(np.sqrt(width/height * nblocks)))
        nrows = int(np.ceil(nblocks / ncols))
        ncols = nblocks if nrows == 1 else ncols

        figure = plt.figure(figsize=(width, height))

        for i, block in enumerate(blocks):
    
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(block, interpolation='nearest', aspect='auto')
            plt.colorbar()

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename)
        
        if show:
            plt.show()
        
        plt.close(figure)
    
    def update(self, direction, length, verbose=False):
        """ Take a step in the direction, peforming certain sanity checks. """

        assert direction.shape == self.weights.shape
        
        if verbose:
            L2 = (direction ** 2).sum() ** 0.5
            print("Length of direction vector: %.5g." % L2)
            print("Length of the step taken: %.5g." % (length * L2))
            print()
        
        shape_before = self.weights.shape
        self.weights += length * direction
        
        assert shape_before == self.weights.shape


class ContinuousPolicy(Policy):
    
    def __init__(self, sdim=None, udim=None, low=None, high=None,
                       environment=None, weights=None, dist=None,
                       filename=None, *args, **kwargs):
        
        if filename is not None:
            self.load(filename)
            self.compile()
            return
        
        if environment is None:
            
            self.sdim = sdim
            self.udim = udim
            
            self.low = low
            self.high = high
        
        else:
            
            self.sdim = environment.observation_space.shape[0]
            self.udim = environment.action_space.shape[0]
            
            self.low = environment.action_space.low
            self.high = environment.action_space.high

        self.dist = distributions.ArctanGaussian() if dist is None else dist
        self.weights = self.random_weights() if weights is None else BlockyVector(weights)

        self.compile()
    
    def wrap(self, action):
        """ Convert an unnormalized action into a box element. """

        return self.low + (self.high - self.low)*action
        
    def unwrap(self, box_elm):
        """ Convert a box element into a unit cube element. """

        return (box_elm - self.low) / (self.high - self.low)
    
    def WRAP(self, action):
        """ Convert an unnormalized action into a box element. """

        return self.low + (self.high - self.low)*action

    def UNWRAP(self, box_elm):
        """ Convert a box element into a unit cube element. """

        return (box_elm - self.low) / (self.high - self.low)

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
        self.paramlist = theano.function(inputs=[SHIST, UHIST] + WEIGHTS, outputs=PARAMS, on_unused_input='ignore')
        self.logp = theano.function(inputs=[ACTION, SHIST, UHIST] + WEIGHTS, outputs=LOGP, on_unused_input='ignore')
        self.dlogp = theano.function(inputs=[ACTION, SHIST, UHIST] + WEIGHTS, outputs=SCORE, on_unused_input='ignore')
        print("Done.\n")
        
    def params(self, shist, uhist, weights=None):
        
        if weights is None:
            weights = self.weights
        
        shist = np.atleast_2d(shist)
        uhist = np.atleast_2d(uhist)
        
        return self.paramlist(shist, uhist, *weights)

    def sample(self, shist, uhist, weights=None):
        
        params = self.params(shist, uhist, weights)
        
        assert len(params) == self.dist.nparams
        
        for param in params:
            assert len(param) == self.udim

        unitboxed = self.dist.sample(params)
        actionboxed = self.wrap(unitboxed)

        assert not np.any(np.isnan(unitboxed))

        assert np.all(np.zeros_like(unitboxed) <= unitboxed)
        assert np.all(unitboxed <= np.ones_like(unitboxed))

        assert np.all(self.low <= actionboxed)
        assert np.all(actionboxed <= self.high)
        
        return actionboxed

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
    
    def random_weights(self):

        inputdim = 1 + (self.sdim * self.degree) # [1] + concatenated powers
        outputdim = self.udim # each parameter has the same dim as the action
        
        pick_matrix = lambda: np.random.normal(size=(outputdim, inputdim))
        matrix_list = [pick_matrix() for _ in range(self.dist.nparams)]
        
        return BlockyVector(matrix_list)

    def PARAMS(self, SHIST, UHIST, WEIGHTS):
        
        STATE = SHIST[-1, :]
        
        POWERS = [STATE ** (n + 1) for n in range(self.degree)]
        INPUTS = tns.concatenate([[1]] + POWERS)

        return [tns.dot(WEIGHT, INPUTS) for WEIGHT in WEIGHTS]


class FeedForwardPolicy(ContinuousPolicy):
    
    def __init__(self, hidden=[], degree=None, *args, **kwargs):
        
        self.hidden = hidden
        self.degree = 1 if degree is None else degree
        
        super().__init__(*args, **{key: val for key, val in kwargs.items()
                                   if key not in ['hidden', 'degree']})
    
    def random_weights(self):
        
        smemory = 2
        umemory = 2
        
        assert self.dist.nparams == 1
        
        firstsize = (smemory*self.sdim + umemory*self.udim)*self.degree
        lastsize = self.udim # note that we only allow a single parameter
        
        indims = [firstsize] + [self.degree*w for w in self.hidden]
        outdims = self.hidden + [lastsize]
        
        weights = [np.random.normal(size=(outdim, indim))
                   for indim, outdim in zip(indims, outdims)]
        
        return BlockyVector(weights)

    def PARAMS(self, SHIST, UHIST, WEIGHTS):
        
        smemory = 2
        umemory = 2
        
        SLIST = [SHIST[-(t + 1), :] for t in range(smemory)]
        ULIST = [UHIST[-(t + 1), :] for t in range(umemory)]
         
        INPUT = tns.concatenate(SLIST + ULIST)

        X = [INPUT]
        
        for WEIGHT_D in WEIGHTS[:-1]:
            LAYER = tns.concatenate([X[-1] ** n for n in range(self.degree)])
            LINEAR = tns.dot(WEIGHT_D, LAYER)
            X.append(tns.tanh(LINEAR))

        LAYER = tns.concatenate([X[-1] ** n for n in range(self.degree)])
        LINEAR = tns.dot(WEIGHTS[-1], LAYER)
        X.append(LINEAR) # no squashing

        return [X[-1]] # list containing only one parameter vector


if __name__ == '__main__':
    
    pass