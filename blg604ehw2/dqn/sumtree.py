""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np


class SumTree():
    """ Binary heap with the property: parent node is the sum of
    two child nodes. Tree has a maximum size and whenever
    it reaches that, the oldest element will be overwritten
    (queue behaviour). All of the methods run in O(log(n)).

    Arguments
        - maxsize: Capacity of the SumTree

    """
    def __init__(self, maxsize):
        ### YOUR CODE HERE ###
        raise NotImplementedError
        ###       END      ###

    def push(self, priority):
        """ Add an element to the tree and with the given priority.
         If the tree is full, overwrite the oldest element.

        Arguments
            - priority: Corresponding priority value
        """
        ### YOUR CODE HERE ###
        raise NotImplementedError
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
        raise NotImplementedError
        ###       END      ###
        return node

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
        raise NotImplementedError
        ###       END      ###
