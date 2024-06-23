import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


def build_tree(X, y, depth=0, max_depth=5):
    num_samples_per_class = [np.sum(y == i) for i in range(n_classes)]
    predicted_class = np.argmax(num_samples_per_class)
    node = Node(predicted_class=predicted_class)

    if depth < max_depth:
        idx, thr = best_split(X, y)  # function to find the best split
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.feature_index = idx
            node.threshold = thr
            node.left = build_tree(X_left, y_left, depth + 1, max_depth)
            node.right = build_tree(X_right, y_right, depth + 1, max_depth)
    return node

def predict(x, node):
    if node.left is None:
        return node.predicted_class
    if x[node.feature_index] < node.threshold:
        return predict(x, node.left)
    else:
        return predict(x, node.right)


class Node:
    def __init__(self, predicted_class, treatment_effect):
        self.predicted_class = predicted_class
        self.treatment_effect = treatment_effect
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def calculate_treatment_effect(y, t):
    pass

def best_split(X, y, t):
    pass



def build_tree(X, y, t, depth=0, max_depth=5):
    num_samples_per_class = [np.sum(y == i) for i in range(n_classes)]
    predicted_class = np.argmax(num_samples_per_class)
    treatment_effect = calculate_treatment_effect(y, t)  # function to calculate the treatment effect
    node = Node(predicted_class=predicted_class, treatment_effect=treatment_effect)

    if depth < max_depth:
        idx, thr = best_split(X, y, t)  # function to find the best split
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left, t_left = X[indices_left], y[indices_left], t[indices_left]
            X_right, y_right, t_right = X[~indices_left], y[~indices_left], t[~indices_left]
            node.feature_index = idx
            node.threshold = thr
            node.left = build_tree(X_left, y_left, t_left, depth + 1, max_depth)
            node.right = build_tree(X_right, y_right, t_right, depth + 1, max_depth)
    return node
