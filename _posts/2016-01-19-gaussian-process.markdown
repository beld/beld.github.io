---
layout:     post
title:      "Kernel Methods and Gaussian Process 高斯过程"
subtitle:   ""
date:       2016-01-19 22:30:00
author:     "Beld"
header-img: "img/post-bg-engineer.jpg"
tags:
    - Machine Learning
---

##### 前言
在回归和分类的线性模型中，输入变量x通过由可调节参数向量w控制的映射y(x,w)映射到输出y。在学习阶段，训练数据被同时用在参数向量的点估计和这个向量上的后验分布确定。然后训练数据就被丢弃，新输入的预测完全基于学习好的参数向量w。这样的方法还被运用于像神经网络这样的非线性参数模型。然而，有训练数据点或它的一个子集在预测阶段仍然保留并被使用的一类模式识别的技术。比如kNN最邻近算法，把每个新的测试向量分配为训练数据集里距离最近的样本的标签。这些都是基于存储（memory-based）的方法的例子。基于存储的方法把整个训练数据存储起来，用来对未来的数据点进行预测。通常这种方法需要一个用来定义输入空间任意两个向量之间的相似度的度量。这种方法通常“训练”速度很快，但是对测试数据点的预测速度很慢。  

许多线性参数模型可以被转化为一个等价的预测的基础，也是在训练数据点处计算的核函数（kernel function）的线性组合的“对偶表示” Dual Representation。正如我们将看到的那样，对于基于固定非线性特征空间（feature space）映射ϕ(x)的模型来说，核函数由形式为
<center>$$k(x,x') = \phi(x)^T\phi(x') \tag{6.1}$$</center>

常用的核函数有各种不同的形式。许多核函数只是参数的差值的函数，即$$k(x,x′)=k(x−x′)$$，因为这样核函数对于输入空间的平移具有不变性，所以被称为静止核（stationary kernel）。另一种核函数是同质核（homogeneous kernel），也被称为径向基函数（radial basis function），它只依赖于参数之间的距离（通常是欧几里得距离）的大小，即$$k(x,x′)=k(∥x−x′∥)$$。

##### 对偶表示 Dual Representation － Kernelized  Regression
许多回归和分类的线性模型的公式都可以使用核函数自然产生的对偶表示来重写，比如svm支持向量机。这里，我们考虑一个参数通过最小化形式为
<center>$$J(w) = \frac{1}{2}\sum\limits_{n=1}^N\left\{w^T\phi(x_n) - t_n\right\}^2 + \frac{\lambda}{2}w^Tw \tag{6.2}$$</center>
正则化的平方和误差函数来确定线性模型。其中$$\lambda \geq 0$$。如果我们令J(w)关于w的梯度等于0，那么我们看到w的解是向量ϕ(xn)的线性组合的形式，系数是w的形式为
<center>$$w = -\frac{1}{\lambda}\sum\limits_{n=1}^N\{w^T\phi(x_n) - t_n\}\phi(x_n) = \sum\limits_{n=1}^Na_n\phi(x_n) = \Phi^Ta \tag{6.3}$$</center>
的函数，其中Φ是设计矩阵，第$$n^{th}$$行由给出，向量$$a = (a_1,...,a_N)^T$$，且我们定义了
<center>$$a_n = -\frac{1}{\lambda}\{w^T\phi(x_n) - t_n\} \tag{6.4}$$</center>
我们现在不直接对参数向量w进行操作，而是使用参数向量a重新整理最小平方算法，得到一个对偶表示（dual representation。如果我们将$$w = \Phi^Ta$$代入$$J(w)$$，那么可以得到
<center>$$J(a) = \frac{1}{2}a^T\Phi\Phi^T\Phi\Phi^Ta - a^T\Phi\Phi^Tt + \frac{1}{2}t^Tt + \frac{\lambda}{2}a^T\Phi\Phi^Ta \tag{6.5}$$</center>
其中$$t = (t_1,...,t_N)^T$$。我们现在定义Gram矩阵$$K = \Phi\Phi^T$$。它是一个$$N×N$$的对称矩阵，元素为
<center>$$K_{nm} = \phi(x_n)^T\phi(x_m) = k(x_n,x_m) \tag{6.6}$$</center>
使用Gram矩阵，平方和误差函数可以写成
<center>$$J(a) = \frac{1}{2}a^TKKa - a^TKt + \frac{1}{2}t^Tt + \frac{\lambda}a^TKa \tag{6.7}$$</center>
令J(a)关于a的梯度为0，得到我们的解：
<center>$$a = (K + \lambda I_N)^{-1}t \tag{6.8}$$</center>
如果我们把它代入线性回归模型中，对于新的输入x，我们得到了下面预测
<center>$$y(x) = w^T\phi(x) = a^T\Phi\phi(x) = k(x)^T(K + \lambda I_N)^{-1}t \tag{6.9}$$</center>
其中我们定义了向量k(x)，它的元素为kn(x)=k(xn,x)。因此我们看到对偶公式使得最小平方问题的解完全通过核函数k(x,x′)表示。这被称为对偶公式，因为a的解可以被表示为ϕ(x)的线性组合，从而我们可以使用参数向量w恢复出原始的公式。注意，在x处的预测由训练集数据的目标值的线性组合给出（the predicted output is a linear combination of the training outputs）。至此原来线性回归方程的参数w消失，由核函数来表示回归方程，以上方式把基于特征的学习转换成了基于样本的学习，避免了显式地引入特征向量ϕ(x)。如果有很多基的话维度势必会很高，计算内积的花销会很大，有些是无限维的，核函数能绕过高维的内积计算，直接用核函数得到内积。

##### Gaussian Process 高斯过程
