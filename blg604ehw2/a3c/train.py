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


def train_worker(args, globalmodel, optim, envfunc, agentfunc, tc, logger):
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
    #import pdb;pdb.set_trace()
    env = envfunc()
    agent = agentfunc()
    agent.train()
    ### YOUR CODE HERE ###
#     import pdb;pdb.set_trace()
    # Remember Logger has the shared time step value
    
    # Worker should be in a loop that terminates when
    # the shared time step value is higher than the
    # maximum time step.
    # Logger = namedtuple("Logger", "eps_reward best_reward best_model time_steps time")
    tstep = 0
    while logger.time.value < args.maxtimestep:
        agent.zero_grad()
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0.
        time_start = tstep
        for step in range(args.maxlen):
            a = agent.soft_policy(agent.v_wrap(np.array(s)))
            #import pdb;pdb.set_trace()
            s_, r, done, _ = env.step(a.squeeze())
            if tstep - time_start == args.maxlen - 1:
                done = True
            ep_r += r #todo pushpull içinden al
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            if tstep % args.nstep == 0 or done: # update global and assign to local net
                #import pdb;pdb.set_trace()
                agent.synchronize(globalmodel.state_dict())
                agent.push_and_pull(optim, globalmodel, done,
                                    s_, buffer_s, buffer_a, buffer_r, args.gamma, args.beta)
                buffer_s, buffer_a, buffer_r = [], [], []
                if done:  # done and print information
                    t = logger.time.value
                    if logger.eps_reward[t] == None:
                        logger.eps_reward[t] = ep_r
                        logger.best_reward[t] = ep_r
                    else:
                        logger.eps_reward[t] = \
                        (logger.eps_reward[t] * (tc-1) + ep_r)/tc
                    logger.best_reward.append(logger.eps_reward[t] if 
                        logger.eps_reward[t] > logger.best_reward[t] else 
                                              logger.eps_reward[t-1])
                    logger.time.value += 1
                    break
            s = s_
            tstep += 1

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

    #raise NotImplementedError

    ###       END      ###