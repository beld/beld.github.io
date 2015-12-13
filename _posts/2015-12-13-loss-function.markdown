---
layout:     post
title:      "Loss Function 损失函数"
subtitle:   "a unifying view"
date:       2015-12-13 00:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---

> Loss function is used to measure the degree of fit.

机器学习狭义地概括来讲，就是找到一个决策函数的形式，然后优化其参数，主要通过最大似然或者最小化损失函数。
损失函数就是用来度量模型的拟合程度。

1. 对分类问题来讲，如何判断分类正确与否可以简单地通过令

$$a^2 + b^2 = c^2$$
