import numpy as np
from theano import tensor as tns
from scipy import special


def sample(a, b, size=None):
    """ Sample from a beta distribution with the given parameters. """
    
    return np.random.beta(a, b, size=size)

def logp(x, a, b, log=np.log, betaln=special.betaln, sum=np.sum):
    """ Numeric log-density according to a Beta distribution. """
    
    return sum((a - 1)*log(x) + (b - 1)*log(1 - x) - betaln(a, b))


def BETALN(A, B):
    """ Symbolically compute the value of the Beta function at (A, B). """
    
    return tns.gammaln(A) + tns.gammaln(B) - tns.gammaln(A + B)

def LOGP(x, a, b):
    """ Symbolic log-density according to a Beta distribution. """
    
    return logp(x, a, b, log=tns.log, betaln=BETALN, sum=tns.sum)


if __name__ == '__main__':

    # assert density integrates to 1.0:
    
    a = np.random.gamma(1, 1)
    b = np.random.gamma(5, 1)
    
    x = np.linspace(0, 1, 1000)
    y = np.exp([logp(xn, a, b) for xn in x])

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
    betasample = sample(a, b, size=samplesize)
    
    empirical = []
    numerical = []
    
    if not np.any(np.isinf(y)):
        midy = 0.5*(maxy + miny)
    else:
        midy = miny
    
    for fraction in np.linspace(.05, .95, 20):

        emp = np.sum(betasample < fraction) / samplesize
        num = np.sum((midx < fraction) * deltax * midy)
        
        empirical.append(emp)
        numerical.append(num)
    
    tol = 0.05 / min(a, b)
    
    assert np.allclose(empirical, numerical, atol=tol, rtol=tol)

    # if you want, do some plotting:

    from matplotlib import pyplot as plt

    plt.figure(figsize=(16, 9))
    plt.title("$a = %.2f$, $b = %.2f$" % (a, b), fontsize=24)
    plt.hist(betasample, bins=50, normed=True)
    plt.plot(x, y, lw=5, alpha=0.5, color="red")
    plt.show()