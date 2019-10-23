import pandas as pd
from sklearn.datasets import load_breast_cancer

from fastiv import FastIV


def main():
    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]
    feature_names = data["feature_names"]

    df = pd.DataFrame(data=X, columns=feature_names)
    print(df)

    fiv = FastIV(criterion="entropy",
                 min_samples_leaf=50,
                 max_leaf_nodes=8,
                 others_threshold=200)

    features = ['mean radius', 'mean texture']
    iv, iv_dict = fiv.fast_iv(df[features], y)
    print("%s: %s\n %s" % (features, iv, iv_dict))

    df_export = fiv.export(mode="df")
    print(df_export)

    bin_node = fiv.transform(df[features].values)
    print(bin_node)


if __name__ == '__main__':
    main()
