import torch
import numpy as np
from collections import namedtuple

from blg604ehw2.utils import process_state
from blg604ehw2.atari_wrapper import LazyFrames

Hidden = namedtuple("Hidden", "actor critic")


class BaseA3c(torch.nn.Module):
    """ Base class for Asynchronous Advantage Actor-Critic agent.
    This is a base class for both discrete and continuous
    a3c implementations.

    Arguments:
        - network: Neural network with both value and
        distribution heads
    """
    def __init__(self, network):
        super().__init__()
        self.network = network
        ### YOUR CODE HERE ###
        # You may skip
        # TODO: try to init w and b
#         import pdb;pdb.set_trace()
#         for layer in \
#         [module for module in self.network.modules()]:
#             try:
#                 torch.nn.init.normal_(layer.weight, mean=0., std=0.1)
#                 torch.nn.init.constant_(layer.bias, 0.)
        ###       END      ###

    def greedy_policy(self):
        """ Return best action at the given state """
        raise NotImplementedError

    def soft_policy(self):
        """ Return a sample from the distribution of
        the given state
        """
        raise NotImplementedError

    def loss(self, transitions, last_state, is_terminal, gamma, beta):
        """ Perform gradient calculations via backward
        operation for actor and critic loses.

        Arguments:
            - transitions: List of past n-step transitions
            that includes value, entropy, log probability and
            reward for each transition
            - last_state: Next state agter the given transitions
            - is_terminal: True if the last_state is a terminal state
            - gamma: Discount rate
            - beta: Entropy regularization constant
        """

        ### YOUR CODE HERE ###

        # Transtions can be either
        #   - reward, value, entropy, log probability
        #   of the states and actions
        #   - state, action
        #   from the bootstrap buffer
        #   First one is suggested!

        raise NotImplementedError

        ###       END      ###
        return actor_loss, critic_loss

    def synchronize(self, state_dict):
        """ Synchronize the agent with the given state_dict """
        self.load_state_dict(state_dict)

    def global_update(self, opt, global_agent):
        """ Update the global agent with the agent's gradients
        In order to use this method, backwards need to called beforehand
        """
        if next(self.parameters()).is_shared():
            raise RuntimeError(
                "Global network(shared) called global update!")
        for global_p, self_p in zip(global_agent.parameters(), self.parameters()):
            if global_p.grad is not None:
                continue
            else:
                global_p._grad = self_p.grad
        opt.step()

    def zero_grad(self):
        """ Clean the gradient buffers """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @property
    def device(self):
        """ Return device name """
        return self._device

    @device.setter
    def device(self, value):
        """ Set device name and the model's
         device.
        """
        super().to(value)
        self._device = value


class ContinuousA3c(BaseA3c):
    """ Continuous action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist, clip_low=-1, clip_high=1):
        """ Return best action at the given state """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def soft_policy(self, action, clip_low=-1, clip_high=1):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###


class DiscreteA3c(BaseA3c):
    """ Discrete action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist):
        """ Return best action at the given state """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def soft_policy(self, action):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###
