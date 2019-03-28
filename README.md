# Optimal-Control-via-Neural-Networks

This is the code repository for ICLR 2019 Paper [Optimal Control Via Neural Networks: A Convex Approach](https://openreview.net/forum?id=H1MW72AcK7)

Authors: Yize Chen*, Yuanyuan Shi* and Baosen Zhang, University of Washington

## Introduction
Control of complex systems involves both system identification and controller design. Deep neural networks
have proven to be successful in many identification tasks, however, from model-based control perspective, these
networks are difficult to work with because they are typically nonlinear and nonconvex. Therefore many systems
are still identified and controlled based on simple linear models despite their poor representation capability. In this
paper we bridge the gap between model accuracy and control tractability faced by neural networks, by explicitly
constructing networks that are convex with respect to their inputs. We show that these input convex networks can be
trained to obtain accurate models of complex physical systems. In particular, we design input convex recurrent neural
networks to capture temporal behavior of dynamical systems. Then optimal controllers can be achieved via solving a
convex model predictive control problem. Experiment results demonstrate the good potential of the proposed input
convex neural network based approach in a two applications: the building HVAC energy management, and Mujoco locomotion tasks respectively.


## Running Building Case
* Environment setup 
** EnergyPlus
* Train Load Forecasting Model
* Run Building Energy Management


## Running Mujoco Case
* Environment setup 
** Mujoco

For the setup of Mujoco environment, we refer to https://github.com/nagaban2/nn_dynamics/blob/master/docs/installation.md

** rllab


* Model Training


Contact: yizechen@uw.edu, yyshi@uw.edu
