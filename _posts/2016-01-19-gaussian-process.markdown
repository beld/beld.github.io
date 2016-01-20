---
layout:     post
title:      "Kernel Methods and Gaussian Process 核与高斯过程"
subtitle:   ""
date:       2016-01-19 22:30:00
author:     "Beld"
header-img: "img/post-bg-engineer.jpg"
tags:
    - Machine Learning
---

##### 前言
在回归和分类的线性模型中，输入变量$$x$$通过由可调节参数向量$$w$$控制的映射$$y(x,w)$$映射到输出$$y$$。在学习阶段，训练数据被同时用在参数向量的点估计和这个向量上的后验分布确定。然后训练数据就被丢弃，新输入的预测完全基于学习好的参数向量$$w$$。这样的方法还被运用于像神经网络这样的非线性参数模型。然而，有训练数据点或它的一个子集在预测阶段仍然保留并被使用的一类模式识别的技术。比如kNN最邻近算法，把每个新的测试向量分配为训练数据集里距离最近的样本的标签。这些都是基于存储（memory-based）的方法的例子。基于存储的方法把整个训练数据存储起来，用来对未来的数据点进行预测。通常这种方法需要一个用来定义输入空间任意两个向量之间的相似度的度量。这种方法通常“训练”速度很快，但是对测试数据点的预测速度很慢。  

许多线性参数模型可以被转化为一个等价的预测的基础，也是在训练数据点处计算的核函数（kernel function）的线性组合的“对偶表示” Dual Representation。正如我们将看到的那样，对于基于固定非线性特征空间（feature space）映射$$ϕ(x)$$的模型来说，核函数由形式为
<center>$$k(x,x') = \phi(x)^T\phi(x') \tag{6.1}$$</center>

常用的核函数有各种不同的形式。许多核函数只是参数的差值的函数，即$$k(x,x′)=k(x−x′)$$，因为这样核函数对于输入空间的平移具有不变性，所以被称为静止核（stationary kernel）。另一种核函数是同质核（homogeneous kernel），也被称为径向基函数（radial basis function），它只依赖于参数之间的距离（通常是欧几里得距离）的大小，即$$k(x,x′)=k(∥x−x′∥)$$。

##### 对偶表示 Dual Representation － Kernelized  Regression
许多回归和分类的线性模型的公式都可以使用核函数自然产生的对偶表示来重写，比如svm支持向量机。这里，我们考虑一个参数通过最小化形式为
<center>$$J(w) = \frac{1}{2}\sum\limits_{n=1}^N\left\{w^T\phi(x_n) - t_n\right\}^2 + \frac{\lambda}{2}w^Tw \tag{6.2}$$</center>
正则化的平方和误差函数来确定线性模型。其中$$\lambda \geq 0$$。如果我们令$$J(w)$$关于$$w$$的梯度等于$$0$$，那么我们看到$$w$$的解是向量$$ϕ(xn)$$的线性组合的形式，系数是$$w$$的形式为
<center>$$w = -\frac{1}{\lambda}\sum\limits_{n=1}^N\{w^T\phi(x_n) - t_n\}\phi(x_n) = \sum\limits_{n=1}^Na_n\phi(x_n) = \Phi^Ta \tag{6.3}$$</center>
的函数，其中Φ是设计矩阵，第$$n^{th}$$行由给出，向量$$a = (a_1,...,a_N)^T$$，且我们定义了
<center>$$a_n = -\frac{1}{\lambda}\{w^T\phi(x_n) - t_n\} \tag{6.4}$$</center>
我们现在不直接对参数向量$$w$$进行操作，而是使用参数向量$$a$$重新整理最小平方算法，得到一个对偶表示。如果我们将$$w = \Phi^Ta$$代入$$J(w)$$，那么可以得到
<center>$$J(a) = \frac{1}{2}a^T\Phi\Phi^T\Phi\Phi^Ta - a^T\Phi\Phi^Tt + \frac{1}{2}t^Tt + \frac{\lambda}{2}a^T\Phi\Phi^Ta \tag{6.5}$$</center>
其中$$t = (t_1,...,t_N)^T$$。我们现在定义Gram矩阵$$K = \Phi\Phi^T$$。它是一个$$N×N$$的对称矩阵，元素为
<center>$$K_{nm} = \phi(x_n)^T\phi(x_m) = k(x_n,x_m) \tag{6.6}$$</center>
使用Gram矩阵，平方和误差函数可以写成
<center>$$J(a) = \frac{1}{2}a^TKKa - a^TKt + \frac{1}{2}t^Tt + \frac{\lambda}a^TKa \tag{6.7}$$</center>
令$$J(a)$$关于$$a$$的梯度为$$0$$，得到我们的解：
<center>$$a = (K + \lambda I_N)^{-1}t \tag{6.8}$$</center>
如果我们把它代入线性回归模型中，对于新的输入x，我们得到了下面预测
<center>$$y(x) = w^T\phi(x) = a^T\Phi\phi(x) = k(x)^T(K + \lambda I_N)^{-1}t \tag{6.9}$$</center>
其中我们定义了向量$$k(x)$$，它的元素为$$k_n(x) = k(x_n,x)$$。

因此我们看到对偶公式使得最小平方问题的解完全通过核函数$$k(x,x′)$$表示。这被称为对偶公式，因为$$a$$的解可以被表示为$$ϕ(x)$$的线性组合，从而我们可以使用参数向量$$w$$恢复出原始的公式。注意，在$$x$$处的预测由训练集数据的目标值的线性组合给出（the predicted output is a linear combination of the training outputs）。至此原来线性回归方程的参数w消失，由核函数来表示回归方程，以上方式把基于特征的学习转换成了基于样本的学习，避免了显式地引入特征向量$$ϕ(x)$$。如果有很多基的话维度势必会很高，计算内积的花销会很大，有些是无限维的，核函数能绕过高维的内积计算，直接用核函数得到内积。

##### Gaussian Process 高斯过程
我们考虑了形式为$$y(x, w) = w^T\phi(x)$$线性回归模型，其中$$w$$是一个参数向量，$$ϕ(x)$$是一个与输入向量$$x$$相关的固定非线性基函数向量。我们证明了，$$w$$上的先验分布会产生函数$$y(x,w)$$上的一个对应的先验分布。给定一个训练数据集，我们计算$$w$$上的后验概率分布，从而就得到和回归函数的对应的后验概率分布。回归函数反过来（叠加上噪声）表示了对新输入向量$$x$$的一个预测分布$$p(t|x)$$。  
在高斯过程的观点中，我们抛弃参数模型，直接定义函数上的先验概率分布。乍一看来，在函数组成的不可数的无穷空间中对概率分布进行计算似乎很困难。但是，对于一个有限的训练数据集，我们只需要考虑训练数据集和测试数据集的输入$$x_n$$处的函数值即可，因此在实际应用中我们可以在有限的空间中进行计算。

###### 定义
通常来说，高斯过程被定义为函数y(x)上的一个概率分布，使得在任意点集$$x_1,...,x_N$$处计算的$$y(x)$$的值的集合联合起来服从高斯分布。也就是说，高斯过程是一个随机变量的集合，使得这个集合任意有限部分都有一个联合高斯分布。这个集合中随机变量的数量可以是无限的。高斯随机过程的一个关键点是$$N$$个变量$$y_1,...,y_N$$上的联合概率分布完全由二阶统计（即均值和协方差）确定。在大部分应用中，我们关于$$y(x)$$的均值没有任何先验的知识，因此根据对称性，我们令其等于零。这等价于基函数的观点中，令权值$$p(w|α)$$的先验概率分布的均值等于$$0$$。之后，高斯过程的确定通过给定两个$$x$$处的函数值$$y(x)$$的协方差来完成。这个协方差由核函数确定
<center>$$\mathbb{E}[y(x_n)y(x_m)] = k(x_n,x_m) \tag{6.55}$$</center>

###### 如何解决无穷问题？
基本思想就是将无穷的随机变量集合分为一个无穷子集和一个有限子集。
<center>$$x=\left( \begin{matrix} { x }_{ f } \\ { x }_{ i } \end{matrix} \right) \sim \mathcal{N}\left( \left( \begin{matrix} { \mu  }_{ f } \\ { \mu  }_{ i } \end{matrix} \right) ,\left( \begin{matrix} { \Sigma  }_{ f } & { \Sigma  }_{ fi } \\ { \Sigma  }_{ if } & { \Sigma  }_{ i } \end{matrix} \right)  \right) $$</center>
然后通过边缘性质，我们可以得到
<center>$$p(x_f)= \int { p(x_f,x_i)} d x_i = \mathcal{N}(x_f| \mu_f, \Sigma_f)$$</center>

###### 高斯过程采样
和对高斯分布进行采样一样，我们也可以对一个高斯过程进行采样，不过得到的每一个样本都是一个函数。过程如下：

- 选择核函数(协方差函数)，其均值函数默认为常量$$0$$.  

```
kernel = 6;
switch kernel
    case 1; k = @(x,y) 1*x'*y; %Linear
    case 2; k = @(x,y) 1*min(x,y); % Brownian  
    case 3; k = @(x,y) exp(-100*(x-y)'*(x-y)); %squared exponential
    case 4; k = @(x,y) exp(-1*sqrt((x-y)'*(x-y))); %Ornstein-Uhlenbeck
    case 5; k = @(x,y) exp(-1*sin(5*pi*(x-y)).^2); %A periodic GP
    case 6; k = @(x,y) exp(-100*min(abs(x-y),abs(x+y)).^2); %A symmetric GP
end
```
- 选择一部分输入点$$\textbf{x}_{1}^{*},...,\textbf{x}_{M}^{*}$$  

```
x = (-1:0.005:1);
n = length(x);
```
- 计算协方差矩阵$$K$$：$$K_ij=k(\textbf{x}_{i}^{*},\textbf{x}_{j}^{*})  

```
C = zeros(n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = k(x(i),x(j));
    end
end
```
- 产生一个随机高斯向量：$$\textbf{y}_*=\mathcal{N}(\textbf{0},K)$$  

```
rn = randn(n,1);%产生n个0~1之间的随机数,满足正态分布
[u,s,v] = svd(C); %svd分解rn矩阵，s为奇异值矩阵，u为奇异向量.C=usv'
z = u*sqrt(s)*rn; %z为什么这么表示,理论是？？
```
- 画出对应的点集  

```
figure(1);hold on; clf
plot(x,z,'.-');
% axis([0,1,-2,2]);
```

![](https://mqshen.gitbooks.io/prml/content/Chapter6/gaussian/images/gaussian_processes.png)

###### 用高斯过程来回归预测
给定一些输入数据，我们可以用高斯过程来预测新的函数值。假定我们有训练数据$$\textbf{x}_1,...,\textbf{x}_N, \quad \textbf{y}_1,...,\textbf{y}_N$$ 和测试数据 $$\textbf{x}_{1}^{*},...\textbf{x}_{M}^{*}$$。那么联合概率就可以表示为
<center>$$\left( \begin{matrix} \textbf{y} \\ { \textbf{y} }_{ * } \end{matrix} \right) \sim { N }\left( \textbf{0},\left( \begin{matrix} K(X,X) & { K(X,X }_{ * }) \\ K({ X }_{ * },X) & { K(X }_{ * },{ X }_{ * }) \end{matrix} \right)  \right) $$</center>
然后我们需要计算$$p(\textbf{y}^{*}|\textbf{x}^{*},X,\textbf{y})$$。对于单个测试点的情况，我们可以推导出预测分布Predictive Distribution，在此略过。

###### 分类的高斯过程
在分类的概率方法中，我们的目标是在给定一组训练数据的情况下，对于一个新的输入向量，为目标变量的后验概率建模。这些概率一定位于区间$$(0,1)$$中，而一个高斯过程模型做出的预测位于整个实数轴上。然而，我们可以很容易地通过使用一个恰当的非线性激活函数，将高斯过程的输出进行变换来调整高斯过程，使其能够处理分类问题。  
首先考虑一个二分类问题，它的目标变量为$$t \in \{0, 1\}$$。如果我们定义函数$$a(x)$$上的一个高斯过程，然后使用logistic sigmoid函数y=σ(a)进行变换（或者用cumulative Gaussian），那么我们就得到了函数$$y(x)$$上的一个非高斯随机过程，其中$$y∈(0,1)$$。但是，与回归不同的是，目标变量$$t$$上的概率分布是伯努利分布：
<center>$$p(t|a) = \sigma(a)^t(1-\sigma(a))^{1-t} \tag{6.73}$$</center>
![高斯过程先验的样本与logistic sigmoid变幻](https://mqshen.gitbooks.io/prml/content/Chapter6/gaussian/images/classification.png)

###### 用高斯过程来分类预测
我们的目标就是计算出预测分布：
<center>$$p(y_{*}=+1|X,\textbf{y},\textbf{x_{*}}) = \int p(y_{*}|f_*)p(f_{*}|X,\textbf{y},\textbf{x}_*)) df_*$$</center>
其中$$f^{*}$$是latent function, $$p(y_{*}|f_*)$$就是一个sigmoid函数。然后在训练集上，我们对所有的latent variables边缘化就得到：
<center>$$p(f_{*}|X,\textbf{y},\textbf{x_{*}}) = \int p(f_{*}|X,\textbf{x}_*),,\textbf{f}) p(\textbf{f}|X,\textbf{y})df$$</center>
最后，我们需要计算所有latent variables的后验概率：
<center>$$p({ \textbf{f} }|X,{ \textbf{y} })=\frac { p (\textbf{y} | \textbf{f}) p(\textbf{f} | X)}{ p(\textbf{y} | X) } $$</center>
其中，$$p (\textbf{y} | \textbf{f})$$是likelihood似然也就是sigmoid函数，$$p(\textbf{f} | X)$$是prior先验概率，$$p(\textbf{y} | X)$$是normalizer。

但是，似然项并不是一个高斯，这意味着我们不能closed form计算后验。这里有不同的解决办法：

1. 拉普拉斯近似 Laplace approximation
2. Expectation Propagation
3. Variational methods
