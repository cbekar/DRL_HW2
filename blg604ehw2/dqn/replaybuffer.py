"""Replay buffer implemantaions for DQN"""
from collections import deque
from collections import namedtuple
from random import sample as randsample
import numpy as np
import torch
from blg604ehw2.dqn.sumtree import SumTree
from numba import jitclass, int64, float64, bool_, jit, njit
from numba.numpy_support import from_dtype

Transition = namedtuple("Transition", ("state",
                                       "action",
                                       "reward",
                                       "next_state",
                                       "terminal")
                        )

spec_memory = [
    ("__per_e", float64),
    ("__per_a", float64),
    ("__per_b", float64),
    ("__per_b_increment_per_sampling", float64),
    ("__absolute_error_uper", float64),
    ("__tree", SumTree),
]

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
    """ Replay buffer that sample transitions
    according to their priorities. Priority
    value is most commonly td error.

    Arguments:
        - capacity: Maximum size of the buffer
        - min_priority: Values lower than the
        minimum priority value will be clipped
        - max_priority: Values larger than the
        maximum priority value will be clipped
    """

    def __init__(self, capacity):
        ### YOUR CODE HERE ###
        self.__per_e = 0.01
        # Hyperparameter used to make a tradeoff between taking only high priority and sampling randomly
        self.__max_priority = 2
        # Importance-sampling, from initial value increasing to 1
        self.__min_priority = 0.1
        # Increment per_b per sampling step
        self.__per_b_increment_per_sampling = 0.001
        # Clipped abs error
        self.__absolute_error_upper = 1.

        self.__tree = SumTree(capacity)
        ###       END      ###

    def _clip_p(self, p):
        # You dont have to use this
        """ Return clipped priority """
        return min(max(p, self.__min_priority), self.__max_priority)

    def push(self, t, priority):
        #import pdb;pdb.set_trace()
        """ Push the transition with priority """
        ### YOUR CODE HERE ###
        max_priority = np.max(self.__tree.get_priority())
        # max_priority = 0

        # We can't put priority = 0 since this exp will never being taken
        if max_priority == 0:
            max_priority = self.__absolute_error_upper

        # add experience in tree
        self.__tree.push(t, max_priority)
        ###       END      ###

    def sample(self, batch_size):
        """ Return namedtuple of transition that
        is sampled with probability proportional to
        the priority values. """
        ### YOUR CODE HERE ###
        memory_batch = []

        batch_idx, batch_ISWeights = (
            np.empty((batch_size,), dtype=np.int32),
            np.empty((batch_size, 1), dtype=np.float32),
        )

        # Calculate the priority segment
        priority_segment = self.__tree.total_priority() / batch_size

        # Increasing per_b by per_b_increment_per_sampling
        self.__min_priority = np.min(
            [1., self.__min_priority + self.__per_b_increment_per_sampling]
        )  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.__tree.get_priority()) / self.__tree.total_priority()
        max_weight = (p_min * batch_size) ** (-self.__min_priority)

        for i in range(batch_size):
            # A value is uniformly sample from each range
            limit_a, limit_b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(limit_a, limit_b)

            # Experience that correspond to each value is retrieved
            index, priority, state, action, reward, next_state, terminal = self.__tree.get(value)

            # P(j)
            sampling_probabilities = priority / self.__tree.total_priority()

            #  IS = (1/batch_size * 1/P(i))**per_b /max wi == (Batch_size*P(i))**-per_b  /MAX(weight)
            batch_ISWeights[i, 0] = (
                np.power(batch_size * sampling_probabilities, -self.__min_priority)
                / max_weight
            )

            batch_idx[i] = index
            memory_batch.append(Transition(state, action, reward, next_state, terminal))

        return batch_idx, memory_batch, batch_ISWeights
        ###       END      ###
    @property
    def size(self):
        """Return the current size of the buffer"""
        return len(self.__tree.get_all_tree())
    
    def update_priority(self, indices, values):
        """ Update the priority value of the transition in
        the given index
        """
        ### YOUR CODE HERE ###
        values += self.__per_e  # convert to abs and avoid 0
        priorities = np.power(self._clip_p(values), self.__max_priority)

        for tree_index, priority in zip(indices, values):
            self.__tree.update(tree_index, priority)
        ###       END      ###