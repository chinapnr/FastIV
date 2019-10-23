import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .parser import tree2json
from .parser import tree2df


class FastIV(object):
    def __init__(self,
                 criterion="entropy",
                 min_samples_leaf=300,
                 max_leaf_nodes=6,
                 others_threshold=200,
                 ignore_nan=False):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.others_threshold = others_threshold
        self.ignore_nan = ignore_nan

        self.tree = DecisionTreeClassifier(criterion=criterion,
                                           min_samples_leaf=min_samples_leaf,
                                           max_leaf_nodes=max_leaf_nodes)

    def category2numeric(self, s: pd.Series, y) -> np.ndarray:
        x = clean(s, self.others_threshold).values
        x, y = cat2num(x, y)
        return x

    def fast_iv(self, df: pd.DataFrame, y):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("the parameter df in fast_iv should be "
                             "pandas.DataFrame")

        if np.any(np.isnan(y)):
            raise ValueError("y should not include NaN.")

        df = df.copy()
        for column in df.columns:
            if df[column].dtype == np.object_:
                df[column] = self.category2numeric(df[column], y)
            elif np.issubdtype(df[column].dtype, np.number):
                continue
            else:
                raise ValueError("dtypes of df should be numpy.object_ or"
                                 "numpy.number")

        return self.numeric_iv(df.values, y)

    def single_iv(self, X, y, dtype=None):
        if dtype is None:
            dtype = X.dtype
        if dtype == np.number:
            return self.numeric_iv(X, y)
        elif dtype == np.object_:
            return self.category_iv(X, y)

    def category_iv(self, X, y):
        s = pd.Series(X.squeeze())
        X = self.category2numeric(s, y)
        return self.numeric_iv(X, y)

    def __numeric_iv(self, X, y):
        pass

    def numeric_iv(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be numpy.ndarray")

        if X.ndim == 1:
            X = X[:, np.newaxis]
        elif X.ndim != 2:
            raise ValueError("the dim of X should be 1 or 2.")

        X_, y_, self.value_index_ = handle_nan(X, y, return_index=True)
        self.bin_ = np.zeros_like(y)
        self.nan_index_ = ~self.value_index_
        self.tree.fit(X_, y_)
        x_ = self.tree.apply(X_)  # stand for the bin labels
        self.bin_[self.value_index_] = x_
        self.bin_[self.nan_index_] = -1

        if not self.ignore_nan:
            iv, ivi_dict = cal_iv(self.bin_, y)
        else:
            iv, ivi_dict = cal_iv(x_, y_)
        # self.ivi_dict_ = ivi_dict
        return iv, ivi_dict

    def export(self, mode="df", feature_names=None):
        if mode == "df":
            return tree2df(self.tree.tree_, feature_names)
        elif mode == "json":
            return tree2json(self.tree.tree_, feature_names)
        else:
            raise ValueError("mode should be in {df, json}")

    def transform(self, X):
        # TODO maybe not support Cate Feature
        X_, value_index = handle_nan(X, return_index=True)
        bin_ = np.zeros((X.shape[0],))
        x_ = self.tree.apply(X_)
        bin_[value_index] = x_
        bin_[~value_index] = -1

        return bin_


def check_xy(x, y):
    if len(x) != len(y):
        raise ValueError("The length of x and y should be equal.")

    if set(y) not in [{0}, {1}, {0, 1}]:
        raise ValueError("The value of y should be 0 or 1 in xy mode.")

    if isinstance(x, (tuple, list)):
        x = np.array(x)
    if isinstance(y, (tuple, list)):
        y = np.array(y)

    return x, y


def _cal_ivi(yi_1, yi_0, n_1, n_0):
    if yi_1 == 0:
        yi_1 = 1
    if yi_0 == 0:
        yi_0 = 1

    pi_1 = yi_1 / n_1
    pi_0 = yi_0 / n_0

    return (pi_1 - pi_0) * np.log(pi_1 / pi_0)


def cal_pos_rate_dict_by_xy(x, y):
    n_1 = sum(y)
    n_0 = len(y) - n_1
    x_classes = sorted(list(set(x)))
    ivi_dict = {}
    for x_class in x_classes:
        yi = y[x == x_class]
        ivi_dict[x_class] = sum(yi) / len(yi)

    return ivi_dict


def cal_ivi_dict_by_xy(x, y):
    n_1 = sum(y)
    n_0 = len(y) - n_1
    x_classes = sorted(list(set(x)))
    ivi_dict = {}
    for x_class in x_classes:
        yi = y[x == x_class]
        yi_1 = sum(yi)
        yi_0 = len(yi) - yi_1
        ivi_dict[x_class] = _cal_ivi(yi_1, yi_0, n_1, n_0)

    return ivi_dict


def cal_iv(x, y):
    """
        in xy mode, x is a binned label series,
        in y1y0 mode, x is a series of numbers that count `label=1` in the each bin
    """

    x, y = check_xy(x, y)
    ivi_dict = cal_ivi_dict_by_xy(x, y)
    iv = sum(ivi_dict.values())
    return iv, ivi_dict


def handle_nan(X, y=None, return_index=False):
    """delete the row with nan values"""
    index = ~np.isnan(X)
    index = np.all(index, axis=1)

    if y is None:
        ret = (X[index],)
    else:
        ret = (X[index], y[index])

    if return_index:
        return (*ret, index)
    return ret


def clean(s: pd.Series, min_samples=10):
    s = s.fillna(value="nan")
    count = s.value_counts()
    others = set(count[count < min_samples].index.values)
    s = s.map(lambda x: "others" if x in others else x)
    # print(s.value_counts())

    return s


def cat2num(x: np.ndarray, y):
    ivi_dict = cal_pos_rate_dict_by_xy(x, y)
    # print(ivi_dict)
    # ivi_dict["nan"] = np.nan

    x_ = np.frompyfunc(lambda item: ivi_dict[item], 1, 1)(x)
    x_ = x_[:, np.newaxis].astype(np.float)

    return x_, y
