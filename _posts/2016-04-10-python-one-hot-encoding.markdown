---
layout:     post
title:      "Python: labels as float 1-hot encodings"
subtitle:   "Understanding == and broadcasting to a NumPy array"
date:       2016-04-10 01:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Python
---
怎么样将numpy一维数组labels变成1-of-K编码的k维数组呢？Tensorflow教程里用了一行代码：

```
labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
```

这一行做了三件事情：numpy's vector ops, adding a singleton axis, and broadcasting。

首先，要理解$$==$$都干了什么。它可以将整个向量与一个scalar进行一一比较，然后得到一个包含比较结果的bool数组。

```
>>> labels = np.array([1,2,0,0,2])
>>> labels == 0
array([False, False,  True,  True, False], dtype=bool)
>>> (labels == 0).astype(np.float32)
array([ 0.,  0.,  1.,  1.,  0.], dtype=float32)
```

得到bool型数组后，再将其强制转换成float型：False==0 in Python, and True==1。到这里，其实我们已经可以用循环得到每个label的one hot编码的方法：

```
>>> np.array([(labels == i).astype(np.float32) for i in np.arange(3)])
array([[ 0.,  0.,  1.,  1.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  1.]], dtype=float32)
```

 但是这样并没有充分利用numpy的优势，我们可以利用numpy broadcasting直接得到结果。

 我们先来看一下broadcasting的官方描述：
 >>The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python.

NumPy operations are usually done on pairs of arrays on an element-by-element basis. In the simplest case, the two arrays must have exactly the same shape, as in the following example:

```
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([2.0, 2.0, 2.0])
>>> a * b
array([ 2.,  4.,  6.])
```
NumPy’s broadcasting rule relaxes this constraint when the arrays’ shapes meet certain constraints. The simplest broadcasting example occurs when an array and a scalar value are combined in an operation:

```
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
array([ 2.,  4.,  6.])
```
我们再来看一些broadcasting的具体实例：

```
>>> x = np.arange(4)
>>> xx = x.reshape(4,1)
>>> y = np.ones(5)
>>> x = np.arange(4)
>>> x
array([0, 1, 2, 3])
>>> x.shape
(4,)
>>> xx
array([[0],
       [1],
       [2],
       [3]])
>>> y = np.ones(5)
>>> xx.shape
(4, 1)
>>> y
array([ 1.,  1.,  1.,  1.,  1.])
>>> y.shape
(5,)
>>> xx + y
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.,  3.],
       [ 4.,  4.,  4.,  4.,  4.]])

>>> a = np.array([0.0, 10.0, 20.0, 30.0])
>>> b = np.array([1.0, 2.0, 3.0])
>>> a[:, np.newaxis] + b
array([[  1.,   2.,   3.],
       [ 11.,  12.,  13.],
       [ 21.,  22.,  23.],
       [ 31.,  32.,  33.]])
```

现在```labels.shape = (5,)```，我们将它增加一个axis，reshape为(5,1)，然后就可以利用broadcasting。

```
>>> np.arange(3) == labels[:,None]
array([[False,  True, False],
       [False, False,  True],
       [ True, False, False],
       [ True, False, False],
       [False, False,  True]], dtype=bool)
>>> (np.arange(3) == labels[:,None]).astype(np.float32)
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  1.]], dtype=float32)
```
Broadcasting用C实现loop循环，而不是在python里，所以是个强有力的工具。
