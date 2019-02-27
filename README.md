REINFORCE tutorial
=================

This repository contains a collection of [scripts](code/) and [notes](pdfs/) that explain the basics of the so-called REINFORCE algorithm, a method for estimating the derivative of an expected value with respect to the parameters of a distribution.

The method was introduced into the reinforcement learning literature by Ronald J. Williams in ["Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) (_Machine Learning_, 1992) but has earlier precedents.

This repository was created to provide some background material for a talk I gave 6 March 2017 at the Berlin machine learning meet-up. The [slides](pdfs/slides.pdf) from the talk are also available here, although they are not completely self-explanatory.

I have also included a few theoretical notes which explain various aspects of REINFORCE, Trust Region Policy Optimization, and other policy gradients methods:

 * ["A Few Observations About Policy Gradient Approximations"](pdfs/A_Few_Observations_About_Policy_Gradient_Approximations.pdf) contains an introductory description of the REINFORCE method;
 * ["Policy Exploration without Back-Looking Terms"](pdfs/Policy_Exploration_without_Back-Looking_Terms.pdf) explains a term-dropping trick that reduces the variance of the gradient estimate without changing its mean;
 * ["A Minimal Working Example of Empirical Gradient Ascent"](pdfs/A_Minimal_Working_Example_of_Empirical_Gradient_Ascent.pdf) explicitly computes the distribution and mean of the gradient estimate in a simple example;
 * ["Policy Exploration in a Cold Universe"](pdfs/Policy_Exploration_in_a_Cold_Universe.pdf) illustrates how the REINFORCE algorithm deals with the exploration/exploitation trade-off in a particularly malicious case.
 * ["Is Randomization Necessary?"](pdfs/Is_Randomization_Necessary.pdf) explains why stochastic policies may be better than deterministic when the policy class isn't convex.

These papers were originally written for internal use in my company, the robot software company [micropsi industries](http://www.micropsi-industries.com/), but are now freely available.