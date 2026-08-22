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

    def pred(self, x):
        """Predict the value for a single individual x."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

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

    def pred(self, x):
        """Predict the value for a single individual x."""
        return self.value

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

    def update_predict(self):
        """Set self.predict to an efficient, vectorized prediction
        function based on the indicator functions of the leaves.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0)

    def pred(self, x):
        """Predict the value for a single individual x."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return the (min, max) of a 1D numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly select a feature and threshold to split node on."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def possible_thresholds(self, node, feature):
        """Return all candidate thresholds for a feature at this node."""
        values = np.unique(
            (self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return the best threshold and its average Gini impurity
        for a single feature, tested against every possible threshold
        at once (fully vectorized, no explicit loop).
        """
        feature_values = self.explanatory[:, feature][node.sub_population]
        targets = self.target[node.sub_population]
        thresholds = self.possible_thresholds(node, feature)
        classes = np.unique(targets)

        is_left = feature_values[:, None] > thresholds[None, :]
        is_class = targets[:, None] == classes[None, :]

        Left_F = is_left[:, :, None] & is_class[:, None, :]
        Right_F = (~is_left)[:, :, None] & is_class[:, None, :]

        left_class_counts = np.sum(Left_F, axis=0)
        right_class_counts = np.sum(Right_F, axis=0)

        left_size = np.sum(left_class_counts, axis=1)
        right_size = np.sum(right_class_counts, axis=1)

        gini_left = 1 - np.sum(
            (left_class_counts / left_size[:, None]) ** 2, axis=1)
        gini_right = 1 - np.sum(
            (right_class_counts / right_size[:, None]) ** 2, axis=1)

        n = feature_values.size
        gini_avg = ((left_size / n) * gini_left +
                    (right_size / n) * gini_right)

        best_idx = np.argmin(gini_avg)
        return thresholds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Return the (feature, threshold) pair with the smallest
        average Gini impurity, among all features.
        """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the given explanatory and target
        data.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {
                self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Recursively split a node into two children (leaf or node)."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold)
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold)

        is_left_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(left_population) < self.min_pop or
            np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(right_population) < self.min_pop or
            np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a Leaf child with the most represented target class."""
        values, counts = np.unique(
            self.target[sub_population], return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a new internal Node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return the accuracy of the tree's predictions on test data."""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) \
            / test_target.size
