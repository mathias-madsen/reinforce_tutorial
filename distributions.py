import numpy as np
from theano import tensor as tns
from scipy.special import gammaln, betaln


def BETALN(A, B):
    """ Symbolically compute the value of the Beta function at (A, B). """

    return tns.gammaln(A) + tns.gammaln(B) - tns.gammaln(A + B)


class Beta(object):
    
    def __init__(self):
        
        self.low = 0.0
        self.high = 1.0
        
        self.nparams = 2
    
    def __repr__(self):
        
        return "Beta()"
    
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


class ArctanGaussian(object):
    
    def __init__(self, low=None, high=None):
        
        self.low = 0.0 if low is None else low
        self.high = 1.0 if high is None else high
        
        self.nparams = 2

    def __repr__(self):
        
        return "ArctanGaussian()"
    
    def sample(self, params, size=None):
        """ Sample from a beta distribution with the given parameters. """
        
        mu = params[0]
        sigma = params[1] + 1e-20
        
        if size is None:
            assert len(set(len(param) for param in params)) == 1 # all == size
        
        gaussian = np.random.normal(loc=mu, scale=np.abs(sigma), size=size)
        
        return self.squash(gaussian)
    
    def LOGP(self, x, params):
        """ Symbolic log-density according to a Beta distribution. """
        
        mu = params[0]
        sigma = params[1]
        
        g = self.UNSQUASH(x)
        square = (g - mu)**2 / sigma**2
        norm = tns.log(2 * np.pi * sigma**2)
        
        return -0.5 * tns.sum(square + norm)

    def logp(self, x, params):
        """ Numeric log-density according to a Beta distribution. """
        
        mu = params[0]
        sigma = params[1] + 1e-20
        
        g = self.unsquash(x)
        square = (g - mu)**2 / sigma**2
        norm = np.log(2 * np.pi * sigma**2)
        
        return -0.5 * np.sum(square + norm)
    
    def squash(self, sample):
        """ Force a sample from the native sample space into the unit box. """
        
        return 0.5 + np.arctan(sample)/np.pi

    def unsquash(self, unit_box_sample):
        """ Perform the inverse of the boxing operation. """
        
        return np.tan(np.pi * (unit_box_sample - 0.5))

    def SQUASH(self, sample):
        """ Perform the boxing operation symbolically (see .box). """
        
        return 0.5 + tns.arctan(sample)/np.pi

    def UNSQUASH(self, unit_box_sample):
        """ Perform the unboxing operation symbolically (see .unbox). """
        
        return tns.tan(np.pi * (unit_box_sample - 0.5))


class NoisyArctan(ArctanGaussian):
    
    def __init__(self, sigma=None):
        
        self.nparams = 1
        self.sigma = 0.1 if sigma is None else sigma
    
    def __repr__(self):
        
        return "NoisyArctan(sigma=%s)" % str(self.sigma)

    def sample(self, params, size=None):
        """ Sample from a beta distribution with the given parameters. """
        
        gaussian = np.random.normal(loc=params[0], scale=self.sigma, size=size)
        
        return self.squash(gaussian)
    
    def LOGP(self, x, params):
        """ Symbolic log-density according to a Beta distribution. """
        
        square = (self.UNSQUASH(x) - params[0])**2 / self.sigma**2
        norm = tns.log(2 * np.pi * self.sigma**2)
        
        return -0.5 * tns.sum(square + norm)

    def logp(self, x, params):
        """ Numeric log-density according to a Beta distribution. """
        
        square = (self.unsquash(x) - params[0])**2 / self.sigma**2
        norm = np.log(2 * np.pi * sigma**2)
        
        return -0.5 * np.sum(square + norm)


if __name__ == '__main__':
    
    from matplotlib import pyplot as plt

    arctangauss = ArctanGaussian()
    n = 10000
    
    y = arctangauss.sample([0, .01], size=n)
    
    plt.hist(y, bins=40)
    plt.show()
    
    x = np.random.normal(loc=0, scale=0.01, size=n)

    plt.hist(0.5 + np.arctan(x)/np.pi, bins=40)
    plt.show()

    plt.hist(arctangauss.squash(x), bins=40)
    plt.show()

    # check that the normalization operation actually does what it says:

    arctangauss = ArctanGaussian()
    mu, sigma = np.zeros(5), np.ones(5)
    
    for i in range(100):
    
        x = np.random.normal(loc=mu, scale=sigma)
        Tx = arctangauss.squash(x)
        x_reconstructed = arctangauss.unsquash(Tx)
    
        assert np.allclose(x, x_reconstructed)
        assert np.all(0 <= Tx) and np.all(Tx <= 1)
    
    for i in range(100):
    
        Tx = arctangauss.sample(mu, sigma)

        assert 0 < Tx and Tx < 1

    # assert density integrates to 1.0:
    
    beta = Beta()
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

    plt.figure(figsize=(16, 9))
    plt.title("$a = %.2f$, $b = %.2f$" % params, fontsize=24)
    plt.hist(sample, bins=50, normed=True)
    plt.plot(x, y, lw=5, alpha=0.5, color="red")
    plt.show()