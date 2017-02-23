import numpy as np


class PolynomialRegressor(object):
    
    def __init__(self, sdim, degree=3):
        
        self.degree = degree
        self.weights = np.zeros((self.degree + 1) * sdim)
    
    def __repr__(self):
        
        return "<%s: degree=%s>" % (self.__class__.__name__, self.degree)
    
    def design(self, states):
        """ For the design matrix (matrix of input vectors) from the states. """

        statearray = np.array(states)
        powers = [statearray ** n for n in range(self.degree + 1)]
        
        return np.concatenate(powers, axis=1)
    
    def predict(self, states):
        
        return self.design(states).dot(self.weights)
        
    def error(self, states, values):
        
        return np.mean((self.predict(states) - values) ** 2)
    
    def fit(self, states, values, caution=0.01, verbose=False):
        
        if verbose:
            error = self.error(states, values)
            print("Fitting baseline (prior error: %.3f) . . ." % error)
        
        inputs = self.design(states)
        solution, residuals, rank, sngrts = np.linalg.lstsq(inputs, values)

        assert solution.shape == self.weights.shape

        self.weights = caution*self.weights + (1 - caution)*solution

        if verbose:
            error = self.error(states, values)
            print("Done (posterior error: %.3f).\n" % error)


if __name__ == '__main__':
    
    sdim = 3
    samples = 100
    maxdegree = 7
    
    A = np.random.normal(size=sdim)
    B = np.random.normal()
    
    states = np.random.normal(size=(samples, sdim))
    values = states.dot(A.T) + B
    
    print(" Fitting to linear data: ".center(46, "="))
    print("")
    
    for d in range(maxdegree):
        
        print("Polynomial function approximator of degree %s:" % d)
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

        approximator = PolynomialRegressor(sdim, degree=d)
        approximator.fit(states, values, verbose=True)
    
    print()

    states = np.random.normal(size=(samples, sdim))
    values = np.sin(states.dot(A.T)) + np.exp(states).dot(A.T)
    
    print(" Fitting to nonlinear data: ".center(46, "="))
    print("")
    
    for d in range(maxdegree):
        
        print("Polynomial function approximator of degree %s:" % d)
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

        approximator = PolynomialRegressor(sdim, degree=d)
        approximator.fit(states, values, verbose=True)
    
    print()