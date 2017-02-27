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


class PolynomialRegressor(Regressor):
    
    def __init__(self, sdim, degree=3):
        
        self.sdim = sdim
        self.degree = degree
        self.params = np.zeros(1 + self.degree*self.sdim)
        
    def __repr__(self):
        
        return "<%s: degree=%s>" % (self.__class__.__name__, self.degree)
    
    def design(self, states):
        """ For the design matrix (matrix of input vectors) from the states. """
        
        states = np.array(states)
        samples, sdim = states.shape

        assert self.sdim == sdim
        
        ones = np.ones((samples, 1))
        
        if self.degree < 1:
            return ones

        statepowers = [states ** (n + 1) for n in range(self.degree)]
        powermatrix = np.concatenate(statepowers, axis=1)
        
        return np.concatenate([ones, powermatrix], axis=1)
    
    def predict(self, states):
        
        return self.design(states).dot(self.params)
    
    def MLE(self, states, values):
        
        inputs = self.design(states)
        solution, residuals, rank, sngrts = np.linalg.lstsq(inputs, values)
        
        return solution


class PolynomialTemporalRegressor(Regressor):
    
    def __init__(self, sdim, degree=3, timedegree=3):
        
        self.sdim = sdim
        self.degree = degree
        self.timedegree = timedegree
        self.params = np.zeros(1 + self.degree*self.sdim + self.timedegree)
    
    def __repr__(self):
        
        return "<%s: degree=%s>" % (self.__class__.__name__, self.degree)
    
    def design(self, states):
        """ For the design matrix (matrix of input vectors) from the states. """
        
        states = np.array(states)
        samples, sdim = states.shape
        
        assert self.sdim == sdim
        
        ones = [np.ones((samples, 1))]
        
        timecolumn = np.arange(samples).reshape((samples, 1))
        timepowers = [timecolumn ** (n + 1) for n in range(self.timedegree)]
        
        statepowers = [states ** (n + 1) for n in range(self.degree)]

        if self.degree == 0 and self.timedegree == 0:
            return np.concatenate(ones, axis=1)

        elif self.degree == 0 and self.timedegree > 0:
            return np.concatenate(ones + timepowers, axis=1)

        elif self.degree > 0 and self.timedegree == 0:
            return np.concatenate(ones + statepowers, axis=1)

        else:
            return np.concatenate(ones + statepowers + timepowers, axis=1)

    def predict(self, states):
        
        return self.design(states).dot(self.params)
    
    def MLE(self, states, values):
        
        inputs = self.design(states)
        solution, residuals, rank, sngrts = np.linalg.lstsq(inputs, values)
        
        return solution


if __name__ == '__main__':
    
    sdim = 3
    samples = 100
    maxdegree = 3
    maxtimedegree = 3
    
    A = np.random.normal(size=sdim)
    B = np.random.normal()
    
    states = np.random.normal(size=(samples, sdim))

    linear = states.dot(A.T) + B
    nonlinear = np.sin(states.dot(A.T)) + np.exp(states).dot(A.T)
    
    for values, datatype in zip([linear, nonlinear], ['linear', 'nonlinear']):
    
        print((" Fitting to %s data: " % datatype).center(46, "="))
        print("")
        print("")
    
        for d in range(maxdegree):

            print("Polynomial function approximator of degree %s):" % d)
            print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

            approximator = PolynomialRegressor(sdim, degree=d)
            approximator.fit(states, values, verbose=True)

            for t in range(maxtimedegree):
        
                print("Temporal-polynomial function approximator of degree (%s, %s):" % (d, t))
                print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

                approximator = PolynomialTemporalRegressor(sdim, degree=d, timedegree=t)
                approximator.fit(states, values, verbose=True)

            print()

        print()
