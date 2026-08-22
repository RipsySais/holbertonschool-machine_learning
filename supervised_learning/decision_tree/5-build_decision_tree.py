#!/usr/bin/env python3
"""Module that defines Node, Leaf and Decision_Tree classes."""

import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node with its feature, threshold and children."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth among the nodes below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count the nodes below this node.

        If only_leaves is True, count only the leaves.
        Otherwise, count this node and all nodes below it.
        """
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Add a prefix to display the left child in the tree string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add a prefix to display the right child in the tree string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        return self.left_child.get_leaves_below() + \
            self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """Recursively compute the lower and upper bounds of each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child is self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute the indicator function of the node.

        The indicator function returns, for a 2D numpy array of
        individuals, a 1D boolean array telling whether each
        individual falls within the bounds of this node.
        """
        def is_large_enough(x):
            """Return True where all features are > the lower bounds."""
            return np.all(np.array(
                [x[:, key] > self.lower[key]
                 for key in list(self.lower.keys())]), axis=0)

        def is_small_enough(x):
            """Return True where all features are <= the upper bounds."""
            return np.all(np.array(
                [x[:, key] <= self.upper[key]
                 for key in list(self.upper.keys())]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def __str__(self):
        """Return a string representation of the node and its subtree."""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            s = (f"-> node [feature={self.feature}, "
                 f"threshold={self.threshold}]\n")
        s += self.left_child_add_prefix(self.left_child.__str__())
        s += self.right_child_add_prefix(self.right_child.__str__())
        return s[:-1]


class Leaf(Node):
    """Represents a leaf (terminal node) of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf with its predicted value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1, since a leaf counts as a single node."""
        return 1

    def get_leaves_below(self):
        """Return a list containing only this leaf."""
        return [self]

    def update_bounds_below(self):
        """Do nothing: a leaf has no children to update."""
        pass

    def __str__(self):
        """Return a string representation of the leaf."""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the Decision_Tree with its hyperparameters."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the tree.

        If only_leaves is True, count only the leaves.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the whole tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute the lower and upper bounds of every node in the tree."""
        self.root.update_bounds_below()
