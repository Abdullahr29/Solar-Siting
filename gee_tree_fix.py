"""Fix for geemap.ml serialization of BEST-FIRST-grown sklearn trees (max_leaf_nodes).

geemap.ml.tree_to_string assumes sklearn stores nodes in pre-order DFS, which is true for
depth-first growth (max_leaf_nodes=None) but FALSE for best-first growth (max_leaf_nodes set).
Best-first trees store nodes in creation order -> geemap emits a mis-ordered/mis-indented
string that GEE's Classifier.decisionTreeEnsemble rejects ("Error parsing line N: expected 8,
got 3").

Fix: physically reorder each tree's node arrays into pre-order DFS (remapping child pointers),
so geemap's proven writer produces a valid string. Saved models are untouched; we reorder a
deepcopy at serialization time via a monkeypatch of geemap.ml.rf_to_strings.
"""
import copy
import numpy as np
from sklearn.tree._tree import Tree
import geemap.ml as _gml

_orig_rf_to_strings = _gml.rf_to_strings


def _preorder(children_left, children_right):
    """Return node ids in pre-order DFS (root, left subtree, right subtree)."""
    order = []
    stack = [0]
    while stack:
        nid = stack.pop()
        order.append(nid)
        if children_left[nid] != children_right[nid]:      # internal node
            stack.append(children_right[nid])              # push right first...
            stack.append(children_left[nid])               # ...so left is popped first
    return np.asarray(order, dtype=np.intp)


def reorder_tree_preorder(estimator):
    """Return a deepcopy of a fitted sklearn tree estimator whose tree_ node storage
    has been reordered into pre-order DFS. Prediction is identical; only storage order
    (and thus geemap's serialization) changes."""
    est = copy.deepcopy(estimator)
    tree = est.tree_
    state = tree.__getstate__()
    nodes = state["nodes"]
    values = state["values"]

    order = _preorder(tree.children_left, tree.children_right)
    if np.array_equal(order, np.arange(len(order))):
        return est                                          # already pre-order; nothing to do

    newid = np.empty(len(order), dtype=np.intp)
    newid[order] = np.arange(len(order))                    # old id -> new id

    new_nodes = nodes[order].copy()
    new_values = values[order].copy()
    for i in range(len(order)):
        l = new_nodes[i]["left_child"]
        r = new_nodes[i]["right_child"]
        if l != -1:
            new_nodes[i]["left_child"] = newid[l]
        if r != -1:
            new_nodes[i]["right_child"] = newid[r]

    n_classes = np.asarray(tree.n_classes, dtype=np.intp)
    new_tree = Tree(tree.n_features, n_classes, tree.n_outputs)
    new_state = dict(state)
    new_state["nodes"] = new_nodes
    new_state["values"] = new_values
    new_tree.__setstate__(new_state)
    est.tree_ = new_tree
    return est


def rf_to_strings_fixed(estimator, feature_names, processes=2, output_mode="INFER"):
    """Drop-in replacement: reorder every sub-tree to pre-order, then delegate to geemap."""
    fixed = copy.copy(estimator)
    fixed.estimators_ = [reorder_tree_preorder(e) for e in estimator.estimators_]
    return _orig_rf_to_strings(fixed, feature_names, processes=processes, output_mode=output_mode)


def apply():
    """Monkeypatch geemap.ml.rf_to_strings so all callers get the fix."""
    _gml.rf_to_strings = rf_to_strings_fixed
