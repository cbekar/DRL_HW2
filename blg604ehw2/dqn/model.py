"""
Deep Q network implementations.

Vanilla DQN and DQN with Duelling architecture,
Prioritized ReplayBuffer and Double Q learning.
"""

import torch
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple

from blg604ehw2.dqn.replaybuffer import UniformBuffer
from blg604ehw2.dqn.replaybuffer import PriorityBuffer
from blg604ehw2.dqn.replaybuffer import Transition
from blg604ehw2.atari_wrapper import LazyFrames
from blg604ehw2.utils import process_state
from blg604ehw2.utils import normalize

class BaseDqn:
    """
    Base class for DQN implementations.

    Both greedy and e_greedy policies are defined.
    Greedy policy is a wrapper for the _greedy_policy
    method.

    Arguments:
        - nact: Number of the possible actions
        int the action space
        - buffer_capacity: Maximum capacity of the
        replay buffer
    """

    def __init__(self, dev, nact, buffer_capacity):
        super().__init__()
        self.nact = nact
        self.buffer_capacity = buffer_capacity
        self._device = dev
        random.seed()
        
    def serialize(self, s):
        if isinstance(s, LazyFrames):
            s = np.array(s, dtype="float32")
            s = s.transpose(2, 0, 1)
            s = normalize(s)
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.device)
        if s.shape[0] != 1:
            s.unsqueeze_(0)
        return s
    
    def greedy_policy(self, state):
        """ Wrapper for the _greedy_policy of the
        inherited class. Performs normalization if
        the state is a LazyFrame(stack of gray images)
        and cast the state to torch tensor with
        additional dimension to make it compatible
        with the neural network.
        """
        ### Optional, You many not use this ###
        #import pdb;pdb.set_trace()
        if isinstance(state, LazyFrames):
            state = np.array(state, dtype="float32")
            state = state.transpose(2, 0, 1)
            state = normalize(state)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.shape[0] != 1:
            state.unsqueeze_(0)
        with torch.no_grad():
            return self.valuenet.forward(state).argmax().item()

    def e_greedy_policy(self, state, epsilon):
        """ Return action from greedy policy
        with the 1-epsilon probability and
        random action with the epsilon probability.
        """
        #import pdb;pdb.set_trace()
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.nact-1)
        else:
            return self.greedy_policy(state)

    def push_transition(self, transition):
        """ Push transition to the replay buffer """
        self.buffer.push(transition)

    def update(self, batch_size, gamma):
        """ Update the model """
        """ Update the valuenet and replace it with the
        targetnet(if period). After the td error is
        calculated for all the batch, priority values
        of the transitions sampled from the buffer
        are updated as well. Return mean absolute td error. 
        """
        
        assert batch_size <= self.buffer.size, "Buffer is not large enough!"

        # This time it is double q learning.
        # Remember the idea behind double q learning.
        ### YOUR CODE HERE ###
        transitions = self.buffer.sample(batch_size)
        #LEARN: *, zip methods
        #import pdb;pdb.set_trace()
        states = torch.stack([e.state.unsqueeze(0) for e in transitions]).float().to(self.device)
        actions = torch.stack([e.action for e in transitions]).float().to(self.device)
        rewards = torch.stack([e.reward for e in transitions]).float().to(self.device)
        next_states = torch.stack([e.next_state.unsqueeze(0) for e in transitions]).float().to(self.device)
        terminals = torch.stack([e.terminal for e in transitions]).byte().to(self.device)
        #import pdb;pdb.set_trace()
        state_action_values = self.batchforward(self.valuenet, states, actions)
        #state_action_values = self.valuenet(states).gather(1, actions.long())
        next_state_values = torch.zeros(batch_size)
        next_state_values = (~terminals).float() * self.batchforward_max(self.target_net, next_states)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + rewards.float()

        loss = torch.nn.MSELoss(reduction='mean')
        output = loss(state_action_values, expected_state_action_values)
        self.opt.zero_grad()
        output.backward()
        for param in self.valuenet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()
        self.target_update_counter += 1
        if self.target_update_counter % self. target_update_period == 0:
            self.target_net.load_state_dict(self.valuenet.state_dict())
        ###       END      ###
        return output.item()           # mean absolute td error
    
    def batchforward_max(self, net, states):
        #import pdb;pdb.set_trace()
        val = torch.from_numpy(np.zeros(states.shape[0])).float().to(self.device)
        for i in range(states.shape[0]):
            val[i] = torch.max(net(states[i]).squeeze())
        return val
        
    def batchforward(self, net, states, actions):
        #import pdb;pdb.set_trace()
        val = torch.from_numpy(np.zeros(states.shape[0])).float().to(self.device)
        for i in range(states.shape[0]):
            val[i] = net(states[i]).squeeze()[actions[i].long()] 
        return val
    
    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        raise NotImplementedError

    @property
    def buffer_size(self):
        """ Return buffer size """
        return self.buffer.size

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

class DQN(BaseDqn, torch.nn.Module):
    """ Vanilla DQN with target network and uniform
    replay buffer. Implementation of DeepMind's Nature
    paper.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_update_period: Number of steps for
        the target network update. After each update
        counter set to zero again (default=100)

    """
    
    def __init__(self, dev, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_update_period=100):
        super().__init__(dev, nact, buffer_capacity)
        self.valuenet = valuenet
        self.target_net = deepcopy(valuenet)
        self.target_update_period = target_update_period
        self.target_update_counter = 0
        self.buffer = UniformBuffer(capacity=buffer_capacity)
        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        ### YOUR CODE HERE ###
        # You may skip this and override greedy_policy directly
        super()._greedy_policy(state)
        ###      END       ###

    def push_transition(self, transition, *args):
        """ Push transition to the replay buffer
            Arguments:
                - transition: Named tuple of (state,
                action, reward, next_state, terminal)
        """
        super().push_transition(transition)

    def update(self, batch_size, gamma):
        """ Update the valuenet and targetnet(if period)
        and return mean absulute td error.Process samples
        sampled from the replay bufferfor q learning update.
        Raise assertion if thereplay buffer is not big
        enough for the batchsize.
        """
        return super().update(batch_size, gamma)

class DuelingDoublePrioritizedDQN(BaseDqn, torch.nn.Module):
    """ DQN implementation with Duelling architecture,
    Prioritized Replay Buffer and Double Q learning. Double
    Q learning idea is implemented with a target network that
    is replaced with the main network at every Nth step.

    Arguments:
        - valuenet: Neural network to represent value
        function.
        - nact: Number of the possible actions
        int the action space
        - lr: Learning rate of the optimization
        (default=0.001)
        - buffer_capacity: Maximum capacity of the
        replay buffer (default=10000)
        - target_replace_period: Number of steps to
        replace value network wtih the target network
        (default=50)

    """

    def __init__(self, dev, valuenet, nact, lr=0.001, buffer_capacity=10000,
                 target_replace_period=50, gamma=0.98):
        super().__init__(dev, nact, buffer_capacity)
        ### YOUR CODE HERE ###
        self.valuenet = valuenet
        self.target_net = deepcopy(valuenet)
        self.target_update_period = target_replace_period
        self.target_update_counter = 0
        if nact == 4:
            s1=1
            s2=8
            s3=0
        else:
            s1=4
            s2=84
            s3=84
        if s3 == 0:
            sh=(buffer_capacity,s1,s2)
        else:
            sh=(buffer_capacity,s1,s2,s3)
        self.buffer = PriorityBuffer(device=self.device,\
                                     shape=sh,capacity=buffer_capacity)
        self.opt = torch.optim.Adam(self.valuenet.parameters(), lr=lr)
        self.gamma = gamma
        ###       END      ###

    def _greedy_policy(self, state):
        """ Return greedy action for the state """
        ### YOUR CODE HERE ###
        super()._greedy_policy(state)
        ###       END      ###
    
    def td_error(self, trans):
        """ Return the td error, predicited values and
        target values.
        """
        # Optional but convenient
        ### YOUR CODE HERE ###
        #import pdb;pdb.set_trace()
        return trans.reward + self.gamma*self.valuenet.forward(trans.next_state).squeeze()[trans.action] - self.valuenet.forward(trans.state).squeeze()[trans.action] #TODO: next_action??
        ###       END      ###

    def push_transition(self, transition):
        """ Push transitoins and corresponding td error
        into the prioritized replay buffer.
        """
        ### YOUR CODE HERE ###
        # Remember Prioritized Replay Buffer requires
        # td error to push a transition. You need
        # to calculate it for the given transition
        self.buffer.push(transition, self.td_error(transition))
        ###       END      ###
        
    def update(self, batch_size, gamma):
        """ Update the valuenet and replace it with the
        targetnet(if period). After the td error is
        calculated for all the batch, priority values
        of the transitions sampled from the buffer
        are updated as well. Return mean absolute td error. 
        """
        return super().update(batch_size, gamma)

