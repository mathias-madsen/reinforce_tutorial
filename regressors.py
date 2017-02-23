import numpy as np


class Regressor(object):
    
    def __init__(self, *args, **kwargs):
        """ Initialize a function approximator. """

        self.params = 0
    
    def __repr__(self):
        
        return "Regressor()"
    
    def design(self, states):
        """ Convert an array of states into an input array of the right shape. """
        
        return states
    
    def predict(self, states):
        """ Predict an array of values based on an array of states. """
        
        raise NotImplementedError
        
    def error(self, states, values):
        
        return np.mean((self.predict(states) - values) ** 2)
    
    def MLE(self, states, values):
        """ Compute the maximum-likelihood parameter settings given the data. """
    
        raise NotImplementedError
        
    def fit(self, states, values, caution=0.01, verbose=False):
        """ Re-estimate the parameters of the Regressor to fit empirical data. """
        
        if verbose:
            error = self.error(states, values)
            print("Fitting baseline (prior error: %.3f) . . ." % error)
        
        solution = self.MLE(states, values)
        self.params = caution*self.params + (1 - caution)*solution

        if verbose:
            error = self.error(states, values)
            print("Done (posterior error: %.3f).\n" % error)


class ZeroRegressor(Regressor):
    
    def __repr__(self):
        
        return "ZeroRegressor()"
    
    def design(self, states):

        return states
    
    def predict(self, states):

        return np.zeros(len(states))
        
    def error(self, states, values):

        return np.mean(values ** 2)
    
    def MLE(self, states, values, caution=0.01, verbose=False):
        
        return 0


class ConstantRegressor(Regressor):
    
    def __repr__(self):
        
        return "<ConstantRegressor>"
    
    def design(self, states):

        return states
    
    def predict(self, states):

        return self.params * np.ones(len(states))
        
    def MLE(self, states, values, caution=0.01, verbose=False):
        
        return np.mean(values)


class PolynomialRegressor(Regressor):
    
    def __init__(self, sdim, degree=3):
        
        self.degree = degree
        self.params = np.zeros((self.degree + 1) * sdim)
    
    def __repr__(self):
        
        return "<%s: degree=%s>" % (self.__class__.__name__, self.degree)
    
    def design(self, states):
        """ For the design matrix (matrix of input vectors) from the states. """

        statearray = np.array(states)
        powers = [statearray ** n for n in range(self.degree + 1)]
        
        return np.concatenate(powers, axis=1)
    
    def predict(self, states):
        
        return self.design(states).dot(self.params)
    
    def MLE(self, states, values):
        
        inputs = self.design(states)
        solution, residuals, rank, sngrts = np.linalg.lstsq(inputs, values)
        
        return solution


if __name__ == '__main__':
    
    sdim = 3
    samples = 100
    maxdegree = 7
    
    A = np.random.normal(size=sdim)
    B = np.random.normal()
    
    states = np.random.normal(size=(samples, sdim))

    linear = states.dot(A.T) + B
    nonlinear = np.sin(states.dot(A.T)) + np.exp(states).dot(A.T)
    
    for values, datatype in zip([linear, nonlinear], ['linear', 'nonlinear']):
    
        print((" Fitting to %s data: " % datatype).center(46, "="))
        print("")
    
        print("Zero function approximator:")
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

        approximator = ZeroRegressor(sdim)
        approximator.fit(states, values, verbose=True)

        print("Constant function approximator:")
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

        approximator = ConstantRegressor(sdim)
        approximator.fit(states, values, verbose=True)

        for d in range(maxdegree):
        
            print("Polynomial function approximator of degree %s:" % d)
            print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

            approximator = PolynomialRegressor(sdim, degree=d)
            approximator.fit(states, values, verbose=True)
    
        print()
