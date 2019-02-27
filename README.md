REINFORCE tutorial
=================

This repository contains a collection of scripts and notes that explain the basics of the so-called REINFORCE algorithm, a method for estimating the derivative of an expected value with respect to the parameters of the underlying sampling distribution.

The method was introduced into the reinforcement learning literature by Ronald J. Williams in ["Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"] (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) (_Machine Learning_, 1992) but has earlier precedents.

This repository was created to provide some background material for a talk I gave 6 March 2017 at the Berlin machine learning meet-up. The ["slides"](slides.pdf) from the talk are also available here.

In addition, I have also included a few theoretical notes here, which explain various aspects of REINFORCE, Trust Region Policy Optimization, and other policy gradients methods:

 * ["A Few Observations About Policy Gradient Approximations"](A Few Observations About Policy Gradient Approximations.pdf) contains an introductory description of the REINFORCE method;
 * ["Policy Exploration without Back-Looking Terms"](Policy Exploration without Back-Looking Terms.pdf) explains a term-dropping trick that reduces the variance of the gradient estimate without changing its mean;
 * ["A Minimal Working Example of Empirical Gradient Ascent"](A Minimal Working Example of Empirical Gradient Ascent.pdf) explicitly computes the distribution and mean of the gradient estimate in a simple example;
 * ["Policy Exploration in a Cold Universe"](Policy Exploration in a Cold Universe.pdf) illustrates how the REINFORCE algorithm deals with the exploration/exploitation trade-off in a particularly malicious case.