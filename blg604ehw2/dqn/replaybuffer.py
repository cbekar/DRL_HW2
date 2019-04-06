"""Replay buffer implemantaions for DQN"""
from collections import deque
from collections import namedtuple
from random import sample as randsample
import numpy as np
import torch

from blg604ehw2.dqn.sumtree import SumTree


Transition = namedtuple("Transition", ("state",
                                       "action",
                                       "reward",
                                       "next_state",
                                       "terminal")
                        )


class BaseBuffer():
    """ Base class for the buffers. Push and sample
    methods need to be override. Initially start with
    an empty list(queue).

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        self.queue = deque(maxlen=capacity)

    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.queue)

    def __len__(self):
        """Return the capacity of the buffer"""
        return self.capacity

    def push(self, transition, *args, **kwargs):
        """Push transition into the buffer"""
        self.queue.append(transition)

    def sample(self, batchsize, *args, **kwargs):
        """Sample transition from the buffer"""
        return randsample(self.queue, batchsize)

class UniformBuffer(BaseBuffer):
    """ Vanilla buffer which was used in the
    nature paper. Uniformly samples transition.

    Arguments:
        - capacity: Maximum size of the buffer
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        ### YOUR CODE HERE ###
        
        ###       END      ###

    def push(self, transition):
        """Push transition into the buffer"""
        ### YOUR CODE HERE ###
        super().push(transition,0,0)
        ###       END      ###

    def sample(self, batchsize):
        """ Return sample of transitions unfiromly
        from the buffer if buffer is large enough
        for the given batch size. Sample is a named
        tuple of transition where the elements are
        torch tensor.
        """
        ### YOUR CODE HERE ###
        return super().sample(batchsize,0,0)
        ###       END      ###

class PriorityBuffer(BaseBuffer):
    """ Replay buffer that sample tranisitons
    according to their prioirties. Prioirty
    value is most commonly td error.

    Arguments:
        - capacity: Maximum size of the buffer
        - min_prioirty: Values lower than the
        minimum prioirty value will be clipped
        - max_priority: Values larger than the
        maximum prioirty value will be clipped
    """

    def __init__(self, capacity, min_prioirty=0.1, max_priority=2):
        super().__init__(capacity)
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def _clip_p(self, p):
        # You dont have to use this
        """ Return clipped priority """
        return min(max(p, self.min_prioirty), self.max_priority)

    def push(self, transition, priority):
        """ Push the transition with priority """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def update_priority(self, indexes, values):
        """ Update the prioirty value of the transition in
        the given index
        """
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###
