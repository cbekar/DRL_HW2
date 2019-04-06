""" Worker functions for training and testing """

import torch
import gym
import numpy as np
from collections import namedtuple
import torch.multiprocessing as mp

from blg604ehw2.utils import LoadingBar

# Hyperparamteres of A3C
A3C_args = namedtuple("A3C_args",
                      """
                        maxtimestep
                        maxlen
                        nstep
                        gamma 
                        lr 
                        beta 
                        device
                      """)


def train_worker(args, globalmodel, optim, envfunc, agentfunc, lock, logger):
    """ Training worker function.
        Train until the maximum time step is reached.
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - optim: Shared optimizer
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
    """
    env = envfunc()
    agent = agentfunc()
    agent.train()
    ### YOUR CODE HERE ###

    # Remember Logger has the shared time step value
    
    # Worker should be in a loop that terminates when
    # the shared time step value is higher than the
    # maximum time step.

    raise NotImplementedError

    ###       END      ###


def test_worker(args, globalmodel, envfunc, agentfunc, lock, logger,
                monitor_path=None, render=False):
    """ Evaluation worker function.
        Test the greedy agent until max time step is
        reached. After every episode, synchronize the
        agent. Loading bar is used to track learning
        process in the notebook.
        
        Arguments:
            - args: Hyperparameters
            - globalmodel: Global(shared) agent for
            synchronization.
            - envfunc: Environment generating function
            - agentfunc: Agent generating function
            - lock: Lock for shared memory
            - logger: Namedtuple of shared objects for
            logging purposes
            - monitor_path: Path for monitoring. If not
            given environment will not be monitored
            (default=None)
            - render: If true render the environment
            (default=False)
    """
    env = envfunc()
    agent = agentfunc()
    agent.eval()
    bar = LoadingBar(args.maxtimestep, "Time step")
    ### YOUR CODE HERE ###

    # Remember to call bar.process with time step and
    # best reward achived after each episode.
    # You may not use LoadingBar (optional).

    # You can include additional logging
    # Remember to change Logger namedtuple and
    # logger in order to do so.

    raise NotImplementedError

    ###       END      ###