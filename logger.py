import numpy as np
from matplotlib import pyplot as plt

import datetime
import os


def makedir(dirpath=None):
    """ Create a directory at results/<now>/ or `dirpath`; return the path. """

    # if no path is given, pick one:
    if dirpath is None:
        now = datetime.datetime.now()
        dirname = now.strftime("%Y_%b_%d_%Hh%M")
        dirpath = os.path.join("results", dirname)

    # if no such directory exists, create:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    return dirpath


def save_args(dirpath, **kwargs):
    """ Save a text file documenting the values of the kwargs. """
    
    logpath = os.path.join(dirpath, "call.txt")
    logfile = open(logpath, "w")
    
    for item in kwargs.items():
        logfile.write("%s=%s\n" % item)
    
    logfile.close()


def print_stats(rewardsums):
    """ Pretty-print some statistical information about the data. """

    mean = np.mean(rewardsums)
    std = np.std(rewardsums)
    
    meantext = "Mean sum-of-rewards per rollout: %.3f ± %.3f." % (mean, std)
    print(meantext, "‾" * len(meantext), "\n")
    
    print("Percentiles of the sums-of-rewards:\n")

    percents = np.linspace(0, 100, 5 + 1)
    percentiles = [np.percentile(rewardsums, p) for p in percents]

    print(" | ".join(("%.0f" % p).center(9) for p in percents))
    print(" | ".join(("%.5g" % p).center(9) for p in percentiles))
    print()


def plot_progress(samples, show=True, filename=None):
    """ Plot the temporal development of a list of lists of numbers. """
    
    plt.figure(figsize=(20, 10))
    
    numepochs = len(samples)
    epochs = np.arange(numepochs)
    
    means = [np.mean(sample) for sample in samples]
    medians = [np.median(sample) for sample in samples]

    for p in np.linspace(5, 50, 10):
        
        top = [np.percentile(sample, 50 + p) for sample in samples]
        bot = [np.percentile(sample, 50 - p) for sample in samples]
        
        plt.fill_between(epochs, bot, top, color="gold", alpha=0.1)
    
    plt.plot(epochs, medians, color="orange", alpha=0.5, lw=5)
    plt.plot(epochs, means, color="blue", alpha=0.5, lw=4)
    
    plt.xlim(-1, numepochs)
    
    plt.xlabel("Training epoch", fontsize=24)
    plt.ylabel("Sum-of-rewards per episode", fontsize=24)
    
    if filename:
        plt.savefig(filename)
    
    if show:
        plt.show()

    plt.close('all')
