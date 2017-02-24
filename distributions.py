import numpy as np
from theano import tensor as tns
from scipy.special import gammaln, betaln


def BETALN(A, B):
    """ Symbolically compute the value of the Beta function at (A, B). """

    return tns.gammaln(A) + tns.gammaln(B) - tns.gammaln(A + B)


class Beta(object):
    
    def sample(self, params, size=None):
        """ Sample from a beta distribution with the given parameters. """
        
        a = params[0]
        b = params[1]
        
        return np.random.beta(a, b, size=size)
    
    def LOGP(self, x, params):
        """ Symbolic log-density according to a Beta distribution. """
        
        a = params[0]
        b = params[1]
        
        return tns.sum((a - 1)*tns.log(x) + (b - 1)*tns.log(1 - x) - BETALN(a, b))
    
    def logp(self, x, params):
        """ Numeric log-density according to a Beta distribution. """
        
        a = params[0]
        b = params[1]
        
        return np.sum((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b))


class Gaussian(object):
    
    def sample(self, params, size=None):
        """ Sample from a beta distribution with the given parameters. """
        
        mu = params[0]
        sigma = params[1]
        
        return np.random.normal(loc=mu, scale=np.abs(sigma), size=size)
    
    def LOGP(self, x, params):
        """ Symbolic log-density according to a Beta distribution. """
        
        mu = params[0]
        sigma = params[1]
        
        square = (x - mu)**2 / sigma**2
        norm = tns.log(2 * np.pi * sigma**2)
        
        return -0.5 * tns.sum(square + norm)

    def logp(self, x, params):
        """ Numeric log-density according to a Beta distribution. """
        
        mu = params[0]
        sigma = params[1]
        
        square = (x - mu)**2 / sigma**2
        norm = np.log(2 * np.pi * sigma**2)
        
        return -0.5 * np.sum(square + norm)


if __name__ == '__main__':

    beta = Beta()
    
    # assert density integrates to 1.0:
    
    params = np.random.gamma(1, 1), np.random.gamma(5, 1)
    
    x = np.linspace(0, 1, 1000)
    y = np.exp([beta.logp(xn, params) for xn in x])

    deltax = x[1:] - x[:-1]
    midx = 0.5*(x[1:] + x[:-1])

    miny = np.min([y[1:], y[:-1]], axis=0)
    maxy = np.max([y[1:], y[:-1]], axis=0)
    
    maxerror = np.sum(deltax * (maxy - miny))
    
    undersum = np.sum(deltax * miny)
    oversum = np.sum(deltax * maxy)
    
    assert undersum < 1
    assert 1 - undersum < maxerror
    
    if not np.any(np.isinf(y)):
    
        assert oversum > 1
        assert oversum - 1 < maxerror

        assert oversum > undersum
        assert oversum - undersum < 2*maxerror
    
    # assert empirical frequencies ~= numerical Glivenko-Cantelli integrals:
    
    samplesize = 10000
    sample = beta.sample(params, size=samplesize)
    
    empirical = []
    numerical = []
    
    if not np.any(np.isinf(y)):
        midy = 0.5*(maxy + miny)
    else:
        midy = miny
    
    for fraction in np.linspace(.05, .95, 20):

        emp = np.sum(sample < fraction) / samplesize
        num = np.sum((midx < fraction) * deltax * midy)
        
        empirical.append(emp)
        numerical.append(num)
    
    tol = 0.05 / min(params)
    
    assert np.allclose(empirical, numerical, atol=tol, rtol=tol)

    # if you want, do some plotting:

    from matplotlib import pyplot as plt

    plt.figure(figsize=(16, 9))
    plt.title("$a = %.2f$, $b = %.2f$" % params, fontsize=24)
    plt.hist(sample, bins=50, normed=True)
    plt.plot(x, y, lw=5, alpha=0.5, color="red")
    plt.show()