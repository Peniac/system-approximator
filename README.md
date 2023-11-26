# system-approximator
If you want to control a system with an RL agent, it is useful to train the latter in a virtual environment. 
With this, one can avoid implications from exploratory RL actions on the production system. 
However, the problem now boils down to approximating the reward signals of the production system w.r.t. state-action pairs. 
This can be solved using a digital twin of the system, i.e., a mathematical model that can generalize the state-action-reward relationship. 

This repo demonstrates how this is possible using a very simple example. 
