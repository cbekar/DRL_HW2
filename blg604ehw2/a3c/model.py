import torch
import numpy as np
from collections import namedtuple

from blg604ehw2.utils import process_state
from blg604ehw2.atari_wrapper import LazyFrames

# def v_wrap(np_array, dtype=np.float32):
#     if np_array.dtype != dtype:
#         np_array = np_array.astype(dtype)
#     return torch.from_numpy(np_array).unsqueeze(0)

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

    def greedy_policy(self, s):
        """ Return best action at the given state """

    def soft_policy(self, s):
        """ Return a sample from the distribution of
        the given state
        """
    
    def v_wrap(self, np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array).unsqueeze(0)

    def push_and_pull(self, opt, global_agent, done, s_, bs, ba, br, gamma, beta):
        """ Perform gradient calculations via backward
        operation for actor and critic loses.

        Arguments:
            - transitions: List of past n-step transitions
            that includes value, entropy, log probability and
            reward for each transition
            - last_state: Next state after the given transitions
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
        if done:
            v_s = 0.              # terminal
        else:
            _, v_s, _ = self.network(self.v_wrap(s_),(None, None))
            v_s = v_s.detach()
        buffer_v_target = []
        for r in br[::-1]:    # reverse buffer r
            v_s = r + gamma * v_s
            buffer_v_target.append(v_s)
        buffer_v_target.reverse()
        opt.zero_grad()
    #def loss(self, transitions, last_state, is_terminal, gamma, beta):
        a_loss, c_loss = \
        self.loss(self.v_wrap(np.vstack(bs)), \
                  self.v_wrap(np.array(ba), dtype=np.int64) \
                  if ba[0].dtype == np.int64 else self.v_wrap(np.vstack(ba)), \
                  self.v_wrap(np.array(buffer_v_target)[:, None]),\
                  beta)
        loss = (a_loss + c_loss).mean() 
        # calculate local gradients and push local parameters to global
        loss.backward()
        self.global_update(opt, global_agent)
        
    def loss(self, S, A, V, b):
        h_a, h_c = (None, None)
        self.train()
        critic_loss, actor_loss = (0., 0.)
        for i in range(S.shape[0]):
            dist, value, (h_a, h_c) = self.network(S[i],(h_a, h_c))
            critic_loss += torch.nn.functional.mse_loss(value, V[i])
            log_prob = dist.log_prob(A[i])
            vnext = value.detach()
#             entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
            #import pdb;pdb.set_trace()
            exp_v = log_prob * (V[i] - vnext) + b * dist.entropy()
            actor_loss -= exp_v
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
        dist, _, _ = self.network.forward(dist,(None,None))
        return dist.type(torch.FloatTensor).mean().clip(clip_low, clip_high)
        ###       END      ###

    def soft_policy(self, s, clip_low=-1, clip_high=1):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        dist, _, _ = self.network.forward(s,(None,None))
        return dist.sample().numpy().clip(clip_low, clip_high)
        ###       END      ###

class DiscreteA3c(BaseA3c):
    """ Discrete action space A3C """
    def __init__(self, network):
        super().__init__(network)

    def greedy_policy(self, dist):
        """ Return best action at the given state """
        ### YOUR CODE HERE ###
        return super().greedy_policy(dist)
        ###       END      ###

    def soft_policy(self, s):
        """ Sample an action  """
        ### YOUR CODE HERE ###
        return super().soft_policy(action)
        ###       END      ###