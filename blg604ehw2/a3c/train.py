""" Worker functions for training and testing """
import time
import torch
import gym
import numpy as np
from collections import namedtuple, deque
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


def train_worker(args, globalmodel, optim, envfunc, agentfunc, t, tc, logger):
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
        #import pdb;pdb.set_trace()
        for step in range(args.maxlen):
            a = agent.soft_policy(agent.serialize(s))
            s_, r, done, _ = env.step(a.squeeze())
            if tstep - time_start == args.maxlen - 1:
                done = True
            ep_r += r #todo pushpull içinden al
            buffer_s.append(agent.serialize(s))
            buffer_a.append(agent.serialize(a))
            buffer_r.append(r)
            if tstep % args.nstep == 0 or done: # update global and assign to local net
                agent.synchronize(globalmodel.state_dict())
                agent.push_and_pull(optim, globalmodel, done,
                                    s_, buffer_s, buffer_a, buffer_r, args.gamma, args.beta)
                buffer_s, buffer_a, buffer_r = [], [], []
                if done:  # done and print information
                    if logger.eps_reward[t] == None:
                        logger.eps_reward[t] = ep_r
                        logger.best_reward[t] = ep_r
                    else:
                        logger.eps_reward[t] = (logger.eps_reward[t] * (tc-1) + ep_r)/tc
                    if logger.eps_reward[t] > logger.best_reward[t]:
                        logger.best_reward.append(logger.eps_reward[t])
                        logger.best_model.synchronize(agent.state_dict())
                    else:
                        logger.best_reward.append(logger.eps_reward[t-1])
                    break
            s = s_
            tstep += 1
            logger.time.value += 1
            logger.time_steps.append(tstep)
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
#     import pdb;pdb.set_trace()
    state = env.reset()
    state = agent.serialize(state)
    reward_sum = 0
    max_rew = -200.
    done = True
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    
    while logger.time_steps[-1] < args.maxtimestep:
        if logger.time_steps[-1] > 0:
            episode_length += 1
            if done:
                cx = torch.zeros(1, 128)
                hx = torch.zeros(1, 128)
            else:
                cx = cx.detach()
                hx = hx.detach()
            #print("start", episode_length)
            with torch.no_grad():
                dist, value, (hx, cx) = logger.best_model.network(state, (hx, cx))
            action = dist.sample().squeeze().numpy()
            state, reward, done, _ = env.step(action)
            done = done or episode_length >= args.maxtimestep
            reward_sum += reward
            # a quick hack to prevent the agent from stucking
            actions.append(action)
            if len(actions) == actions.maxlen:
                done = True
            if done:
                reward_sum = 0
                actions.clear()
                state = env.reset()
            if max_rew < reward_sum:
                max_rew = reward_sum
            state = agent.serialize(state)
            print("train steps", logger.time.value, "test steps" ,episode_length, "max reward", max_rew)
            bar.progress(episode_length, max_rew)
        else:
            time.sleep(1)
    ###       END      ###