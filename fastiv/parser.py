import copy

import numpy as np
import pandas as pd

from .constant import INF


def tree2scopes(tree):
    scopes = {0: [[-INF, INF] for _ in range(tree.n_features)]}
    for i in range(tree.node_count):
        if tree.feature[i] != -2:
            threshold = tree.threshold[i]
            feature = tree.feature[i]
            current_scope = scopes[i]

            left_scope = copy.deepcopy(current_scope)
            left_scope[feature][1] = threshold
            right_scope = copy.deepcopy(current_scope)
            right_scope[feature][0] = threshold

            scopes[tree.children_left[i]] = left_scope
            scopes[tree.children_right[i]] = right_scope

    return scopes


def scopes2df(tree, scopes, feature_names=None):
    """left scope is open while right one is closed"""
    if feature_names is None:
        feature_names = ["f" + str(i) for i in range(tree.n_features)]

    if len(feature_names) != tree.n_features:
        raise ValueError("the length of feature_names is not equal to"
                         " n_features.")

    feature_columns = []
    for feature_name in feature_names:
        feature_columns.append(feature_name + "_low")
        feature_columns.append(feature_name + "_high")

    value_columns = []
    for i in range(tree.n_classes[0]):
        value_columns.append("value_" + str(i))

    columns = feature_columns + value_columns

    data = []
    node = []
    for i in range(tree.node_count):
        if tree.feature[i] == -2:
            node.append(i)
            row = np.ravel(scopes[i])
            row = np.concatenate((row, tree.value[i][0]))
            data.append(row)

    data = np.array(data)
    df = pd.DataFrame(data=data, columns=columns)
    df["node"] = node
    df = df.astype(dtype={value_column: int for value_column in value_columns})

    return df


def tree2df(tree, feature_names=None):
    scopes = tree2scopes(tree)
    df = scopes2df(tree, scopes, feature_names=feature_names)

    return df


def tree2json(tree, feature_names=None):

    if feature_names is None:
        feature_names = ["f" + str(i) for i in range(tree.n_features)]

    if len(feature_names) != tree.n_features:
        raise ValueError("the length of feature_names is not equal to"
                         " n_features.")

    def recursion(node):
        node = int(node)
        feature_index = int(tree.feature[node])
        value = [int(i) for i in tree.value[node][0]]
        class_ = int(np.argmax(value))
        if feature_index == -2:
            return {
                "node": node,
                "is_leaf": True,
                "value": value,
                "class": class_,
            }
        else:
            return {
                "node": node,
                "is_leaf": False,
                "value": value,
                "class": class_,
                "feature": feature_names[feature_index],
                "threshold": float(tree.threshold[node]),
                "left": recursion(tree.children_left[node]),
                "right": recursion(tree.children_right[node]),
            }

    return recursion(0)
