import gym
import numpy as np
import torch
from collections import namedtuple
import torch.multiprocessing as mp 

import gym
from blg604ehw2.atari_wrapper import ClipRewardEnv
from blg604ehw2.atari_wrapper import FrameStack
from blg604ehw2.atari_wrapper import EpisodicLifeEnv
from blg604ehw2.atari_wrapper import WarpFrame
from blg604ehw2.atari_wrapper import ScaledFloatFrame

from blg604ehw2.network import Cnn
from blg604ehw2.network import DiscreteDistHead
from collections import namedtuple
import torch.multiprocessing as mp 
from blg604ehw2.a3c import DiscreteA3c
from blg604ehw2.a3c import A3C_args
from blg604ehw2.network import Network
from blg604ehw2.a3c import SharedAdam
from blg604ehw2.a3c import train_worker
from blg604ehw2.a3c import test_worker
# Breakout Environment
envname = "Breakout-v4"
Logger = namedtuple("Logger", "eps_reward best_reward best_model time_steps time")

# Hyperparameters
breakout_args = A3C_args(
    **dict(
        maxtimestep=40000000,
        maxlen=2000,
        nstep=20,
        gamma=0.98,
        lr=0.00003,
        beta=0.01,
        device="cpu",
    )
)

def breakout_agent():
    feature_net = Cnn(4, 512) # 4 channel size because of the StackFrame buffer
    head_net = DiscreteDistHead(512, 4) # 4 output because of the environment's action space
    network = Network(feature_net, head_net)
    agent = DiscreteA3c(network)
    agent.device = breakout_args.device
    return agent

def breakout_env():
    env = gym.make(envname)
    env = ClipRewardEnv(env)            # Clip the reward between -1 and 1
    env = WarpFrame(env)                # Downsample rgb (210, 160, 3) images to gray images (84, 84)
    env = EpisodicLifeEnv(env)          # Terminate the environment after a live is lost
    env = FrameStack(env, k=4)          # Stack consecutive frames as a single state
    return env


#%%
### Main cell for Breakout ###

N_PROCESSES = mp.cpu_count()

global_agent = breakout_agent()
global_agent.share_memory()
sharedopt = SharedAdam(global_agent.parameters(), lr=breakout_args.lr)

best_agent = breakout_agent()
best_agent.share_memory()

# Try to use one manager
manager = mp.Manager()
logger = Logger(
    manager.list(),
    manager.list(),
    best_agent,
    manager.list(),
    manager.Value("i", 0)
)
logger.time_steps.append(0)
for t in range(N_PROCESSES):
    logger.eps_reward.append(None)
    logger.best_reward.append(None)
lock = mp.Lock()

processes = []

process = mp.Process(target=test_worker,
                     args=(breakout_args, global_agent, breakout_env, breakout_agent, lock, logger))
train_worker(breakout_args, global_agent, sharedopt, breakout_env, breakout_agent, 0, N_PROCESSES, logger)
#test_worker(breakout_args, global_agent, breakout_env, breakout_agent, N_PROCESSES, logger)
"""process.start()
processes.append(process)
for t in range(N_PROCESSES):
    process = mp.Process(target=train_worker,
                         args=(breakout_args, global_agent, sharedopt, breakout_env, breakout_agent, t, N_PROCESSES, logger))
    process.start()
    processes.append(process)
for p in processes:
    p.join()

#%%
# Save the best model's parameters
model_path = "monitor/Breakout/model_state_dict"
torch.save(logger.best_model.state_dict(), model_path)"""