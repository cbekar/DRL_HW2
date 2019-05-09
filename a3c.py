import gym
import numpy as np
import torch
from collections import namedtuple
import torch.multiprocessing as mp 

from blg604ehw2.network import FcNet
from blg604ehw2.network import ContinuousDistHead
from blg604ehw2.network import Network

from blg604ehw2.a3c import ContinuousA3c
from blg604ehw2.a3c import SharedAdam
from blg604ehw2.a3c import train_worker
from blg604ehw2.a3c import test_worker
from blg604ehw2.a3c import A3C_args


# Bipedal Walker environment is similar to Lunar Lander
# State space is a vector of length 24 and there are
# 4 actions
envname = "BipedalWalker-v2"

# Logger is a named tuple of shared lists integer and a model
# It is necessary to have a shared object since it can be used
# by many processes
Logger = namedtuple("Logger", "eps_reward best_reward best_model time_steps time")

# Hyperparameters, again tunning is necessary but optional.
a3c_args = A3C_args(
    **dict(
        maxtimestep=100000,     # Number of time steps for training
        maxlen=600,             # Maximum length of an episode
        nstep=20,               # Bootsrapping length (n-step td)
        gamma=0.98,             # Discount rate
        lr=0.0001,              # Learning rate
        beta=0.01,              # Entropy regularization constant
        device="cpu",           # Device
    )
)

# Agent generating function
def a3c_agent():
    feature_net = FcNet(24)
    head_net = ContinuousDistHead(128, 4)
    network = Network(feature_net, head_net)
    agent = ContinuousA3c(network)
    agent.device = a3c_args.device
    return agent

# Environment generating function
# You can use RewardClip wrapper
def walker_env():
    env =  gym.make(envname)
    return env
    


#%%
### Main cell for Bipedal Walker ###

# Number of training workers
N_PROCESSES = mp.cpu_count()

# Global agent that will be used for synchronization.
global_agent = a3c_agent()
global_agent.share_memory()         # Make sure it is in the shared memory!

# Shared optimizer, since the optimizer has its own parameters
# they need to be in the shared memory as well.
sharedopt = SharedAdam(global_agent.parameters(), lr=a3c_args.lr)

# Another agent for logging purposes
best_agent = a3c_agent()
best_agent.share_memory()

# Logger
# Manager controls another process(server process) to share
# objects between multiple processes via proxies.
# Please read https://docs.python.org/3.7/library/multiprocessing.html
# for more information.
manager = mp.Manager()
logger = Logger(
    manager.list(),
    manager.list(),
    best_agent,
    manager.list(),
    manager.Value("i", 0)
)
logger.time_steps.append(0)
for t in range(a3c_args.maxtimestep):
    logger.eps_reward.append(None)
    logger.best_reward.append(None)
    
# Lock is not necessary
lock = mp.Lock()

# Start by creating a test worker
processes = []
process = mp.Process(target=test_worker,
                     args=(a3c_args, global_agent, walker_env, a3c_agent, lock, logger,\
                           None,False))
# test_worker(a3c_args, global_agent, walker_env, a3c_agent, N_PROCESSES, logger)
# train_worker(a3c_args, global_agent, sharedopt, walker_env, a3c_agent, 0, N_PROCESSES, logger)
process.start()
processes.append(process)

# # Train workers
for t in range(N_PROCESSES):
    process = mp.Process(target=train_worker,
                         args=(a3c_args, global_agent, sharedopt, walker_env, a3c_agent, t, N_PROCESSES, logger))
    process.start()
    processes.append(process)
    
# Wait until all done
for p in processes:
    p.join()

#%%
# Save the best model's parameters
model_path = "monitor/Bipedal/model_state_dict"
torch.save(logger.best_model.state_dict(), model_path)