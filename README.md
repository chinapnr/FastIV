## 使用指南

### 安装
可以选择通过`setup.py`安装或者通过`pip`安装
```
cd fastiv
python setup.py install
```
或者
```
pip install fastiv
```

### 使用

`FastIV`支持便捷的`Information Value (IV)`计算，同时也支持交叉特征的计算。IV计算是通过使用决策树的方式，确定最终分箱的方法，从而计算出相应的IV值。
通过使用交叉的方式，可以确定组合特征的IV值，从中可以选出更有效的特征，加入到模型中。这一方法的思想，借鉴于第四范式的“AutoCross”。其中特征之间的交叉过程，
是通过决策树的分裂来实现的。
同时，我们提供了将决策树按照一定格式输出的接口，可以方便的将分裂节点打印出来。使用该包的调用代码示例：
```
from fastiv import FastIV

fiv = FastIV(criterion="entropy",
             min_samples_leaf=50,
             max_leaf_nodes=8,
             others_threshold=200)

# 选择要交叉的特征
features = ['feature1', 'feature2']

# 计算iv和iv_dict
iv, iv_dict = fiv.fast_iv(df[features], y)

# 以DataFrame格式输出分箱情况
df_export = fiv.export(mode="df")

# 输入特征，返回所属箱对应的节点索引
bins = fiv.transform(df[features].values)
```
完整的代码可以参考`example.py`