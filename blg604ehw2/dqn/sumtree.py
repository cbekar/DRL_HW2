""" Sum Tree implementation for the prioritized
replay buffer.

This class has been purposefully stolen from  https://github.com/FelipeMarcelino/2048-DDQN-PER-Reinforcement-Learning/blob/master/memory.py
My reasoning for this shamelessness is that just in time compiler, ergo low-level optimization for the replay buffer is beyond the scope of this course however it may lead to efficient implementations of the DDPQN, since we have the vanilla DQN as baseline, it is entirely up to us to squeeze the juice out of DDPQN. 
"""
import numpy as np
from numba import jitclass, jit, njit
from numba.numpy_support import from_dtype
from numba import int64, float64, bool_
import torch

# spec_sum_tree = [
#     ("__capacity", int64),
#     ("__data_pointer", int64),
#     ("__tree", float64),
#     ("__state", float64),
#     ("__action", int64),
#     ("__reward", float64),
#     ("__next_state", float64),
#     ("__done", bool_),
# ]

#@jitclass(spec_sum_tree)
class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """
    def __init__(self, device, shape, maxsize):
        ### YOUR CODE HERE ###
        # Pointer to leaf tree
        #import pdb;pdb.set_trace()
        self.__data_pointer = 0
        self.device = device
        # Numbers of leaf nodes that contains experience
        self.__capacity = maxsize

        # Initialize the tree with all nodes equal zero
        # Leaf nodes = capacity
        # Parent nodes = capacity - 1(minus root)
        # Priority tree = 2 * capacity - 1
        self.__tree = torch.from_numpy(np.zeros(2 * maxsize - 1)).float().to(self.device)

        # Initialize experience tree with zeros
        self.__state = torch.from_numpy(np.zeros(shape)).float().to(self.device)
        self.__action = torch.from_numpy(np.zeros(maxsize)).float().to(self.device)
        self.__reward = torch.from_numpy(np.zeros(maxsize)).float().to(self.device)
        self.__next_state = torch.from_numpy(np.zeros(shape)).float().to(self.device)
        self.__done = torch.from_numpy(np.zeros(maxsize)).byte().to(self.device)
        ###       END      ###

    def push(self, t, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        ### YOUR CODE HERE ###
        # Put data inside arrays
        #import pdb;pdb.set_trace()
        self.__state[self.__data_pointer] = t.state
        self.__action[self.__data_pointer] = t.action
        self.__reward[self.__data_pointer] = t.reward
        self.__next_state[self.__data_pointer] = t.next_state
        self.__done[self.__data_pointer] = t.terminal*1

        # Update prioritized tree. Obs: Fill the leaves from left to right
        tree_index = self.__data_pointer + self.__capacity - 1
        self.update(tree_index, priority)

        # Change the data pointer to next leaf
        self.__data_pointer += 1

        # Check if data pointer reaches the maximum capacity, than back to the first index and overwrite data
        if self.__data_pointer >= self.__capacity:
            self.__data_pointer = 0
        ###       END      ###
        
    def get(self, priority):
        """ Return the node with the given priority value.
        Prioirty can be at max equal to the value of the root
        in the tree.

        Arguments
            - priority: Value whose corresponding index
                will be returned.
        """
        ### YOUR CODE HERE ###
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.__tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node
                if priority <= self.__tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    priority -= self.__tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.__capacity + 1

        return (
            leaf_index,
            self.__tree[leaf_index],
            self.__state[data_index],
            self.__action[data_index],
            self.__reward[data_index],
            self.__next_state[data_index],
            self.__done[data_index],
        )
        ###       END      ###
        return node
    
    def total_priority(self):
        """Get total priority's tree"""
        return self.__tree[0]

    def get_priority(self):
        #import pdb;pdb.set_trace()
        t = self.__tree[-self.__capacity:]
        return t

    def get_all_tree(self):
        return self.__tree
    
    def update(self, idx, value):
        """ Update the tree for the given idx with the
        given value. Values are updated via increasing
        the priorities of all the parents of the given
        idx by the difference between the value and
        current priority of that idx.

        Arguments
            - idx: Index of the data(not the tree).
            Corresponding index of the tree can be
            calculated via; idx + tree_size/2 - 1
            - value: Value for the node at pointed by
            the idx
        """
        ### YOUR CODE HERE ###
        # Change new priority score - former priority score
        change = value - self.__tree[idx]
        self.__tree[idx] = value

        # Propagate changes through tree and change the parents
        while idx != 0:
            idx = (idx - 1) // 2  # Round the result to index
            self.__tree[idx] += change
        ###       END      ###
