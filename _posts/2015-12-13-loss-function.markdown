---
layout:     post
title:      "Loss Function 损失函数"
subtitle:   "  -- A unifying view"
date:       2015-12-13 00:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---

> Loss function is used to measure the degree of fit.

机器学习狭义地概括来讲，就是找到一个决策函数的形式，然后优化其参数，主要通过最大似然或者最小化损失函数。
损失函数就是用来度量模型的拟合程度。

##### Classification
对分类问题来讲，可以非常直观地通过$$y⋅f=y⋅{ θ }^{ T }x>0$$为正确，$$y⋅f=y⋅{ θ }^{ T }x<0$$为错误来判断。

**0/1 Loss**: $$ { min }_{ θ }\sum_{ i }{ { L }_{ 0/1 } } \left( { θ }^{ T }x \right) $$.
Non convex and very hard to optimize.

**Hinge Loss**: The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs). Approximate 0/1 loss by $${ min }_{ θ }\sum_{ i }{ { H } } \left( { θ }^{ T }x \right)$$. We define $$H\left( { θ }^{ T }x \right) =max\left( 0,1-y\cdot f \right)$$. Apparently H is small if we classify correctly. The hinge loss provides a relatively tight, convex upper bound on the 0–1 indicator function.

**Logistic Loss**: $${ min }_{ θ }\sum_{ i }{ log\left( 1+exp\left( -y\cdot { θ }^{ T }x \right)  \right)  }$$. This function displays a similar convergence rate to the hinge loss function, and since it is continuous, gradient descent methods can be utilized.

##### Regression
**Square Loss**: $${ min }_{ θ }\sum_{ i }{ { \left\| { y }^{ \left( i \right)  }-{ θ }^{ T }{ x }^{ \left( i \right)  } \right\|  }^{ 2 } }$$. The square loss function is both convex and smooth and matches the 0–1 indicator function. However, the square loss function tends to penalize outliers excessively, leading to slower convergence rates (with regards to sample complexity) than for the logistic loss or hinge loss functions.

##### logistic regression
**Cross Entropy Loss**, also known as *log loss*.
The cross entropy loss is closely related to the Kullback-Leibler divergence between the empirical distribution and the predicted distribution. This function is not naturally represented as a product of the true label and the predicted value, but is convex and can be minimized using stochastic gradient descent methods. The cross entropy loss is ubiquitous in modern deep neural networks.
