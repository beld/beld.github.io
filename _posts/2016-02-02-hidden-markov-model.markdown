---
layout:     post
title:      "隐马尔科夫模型（一）"
subtitle:   ""
date:       2016-02-02 23:11:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---

#### 前言
在数据集里的数据点是独立同分布的假设下，将似然函数表示为在每个数据点处计算的概率分布在所有数据点上的乘积。然而，对于许多应用来说，独立同分布的假设不成立。我们考虑这样的数据集中的一个重要的类型，即顺序数据的数据集。这些数据集通常产生于沿着时间序列进行的测量，例如某个特定位置的连续若干天的降水量测量，或者每天汇率的值，或对于语音识别任务，在连续的时间框架下的声学特征。

区分静止顺序分布和非静止顺序分布是很有用的。在静止分布中，数据会随着时间发生变化，但是生成数据的概率分布保持不变。对于更复杂的非静止分布的情形，生成概率本身会随 着时间变化。这里，我们关注的是静止分布的情形。

对于许多应用来说，我们希望能够在给定时间序列中的前一个观测值的条件下，预测下一个观测值。考虑未来的观测对所有之前的观测的一个一般的依赖关系是不现实的，因为这样一个模型的 复杂度会随着观测数量的增加而无限制地增长。这使得我们要考虑马尔科夫模型（Markov model），其中我们假定未来的预测仅与最近的观测有关，而独立于其他所有的观测。

虽然这样的模型可以计算，但是仍然具有很严重的局限性。通过引入潜在变量，我们可以得到一个更加一般的框架，同时仍然保持计算上的可处理性，这就引出了状态空间模型（state space model）。状态空间模型的最重要的例子，就隐马尔可夫模型（hidden Markov model），其中潜在变量是离散的。这两个模型都使用具有树结构（没有环）的有向图描述，这样就可以使用加-乘算法来高效地进行推断。

#### 马尔科夫模型 Markov Model
用概率的乘积规则来表示观测序列的联合概率分布，形式为
<center>$$p(x_1,...,x_N) = p(x_1)\prod\limits_{n=2}^Np(x_n|x_1,...,x_{n-1}) $$</center>
如果我们现在假设右侧的每个条件概率分布只与最近的一次观测有关，而独立于其他所有之前的观测，那么我们就得到了一阶马尔科夫链（first-order Markov chain）。根据d-划分的性质，给定时刻$$n$$之前的所有观测，我们看到观测$$x_n$$的条件概率分布为
<center>$$p(x_n|x_1,...,x_{n-1}) = p(x_n|x_{n-1}) $$</center>
如果条件概率分布p(xn|xn−1)被限制为相等的，对应于静止时间序列的假设。这样，这个模型被称为同质马尔科夫链（homogeneous Markov chain）。虽然这比独立的模型要一般一些，但是仍然非常受限。一种让更早的观测产生影响的方法是使用高阶的马尔科夫链。$$M$$阶马尔科夫链，其中一个特定的变量依赖于前$$M$$个变量。假设观测是具有$$K$$个状态的离散变量，那么这种模型中参数的数量为$$K^{M}(K−1)$$。 由于这个量随着$$M$$指数增长，因此通常对于大的$$M$$来说，使用这种方法是不实际的。

假设我们希望构造任意阶数的不受马尔科夫假设限制的序列模型，同时能够使用较少数量的自由参数确定。对于每个观测$$x_n$$，我们引入一个对应的潜在变量$$z_n$$（类型或维度可能与观测变量不同）。我们现在假设潜在变量构成了马尔科夫链，得到的图结构被称为状态空间模型（state space model）。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_5.png)
它满足下面的关键的条件独立性质，即给定$$z_n$$的条件下，$$z_{n-1}$$和$$z_{n+1}$$是独立的，从 而
<center>$$z_{n+1} \perp z_{n-1}|z_n$$</center>
这个模型的联合概率分布为
<center>$$p(x_1,...,x_N,z_1,...,z_N) = p(z_1)\left[\prod\limits_{n=2}^Np(z_n|z_{n-1})\right]\prod\limits_{n=1}^Np(x_n|z_n) $$</center>
使用d-划分准则，我们看到总存在一个路径通过潜在变量连接了任意两个观测变量$$x_n$$和$$x_m$$，且这个路径永远不会被阻隔。因此对于观测变量$$x_{n+1}$$来说，给定所有之前的观测，条件概率分布$$p(x_{n+1}|x_1,...,x_n)$$不会表现出任何的条件独立性，因此我们
**对$$x_{n+1}$$的预测依赖于所有之前的观测**。

如果潜在变量是离散的，那么我们得到了隐马尔科夫模型（hidden Markov model）或HMM。注意，HMM中的观测变量可以是离散的或者是连续的，并且可以使用许多不同的条件概率分布进行建模。

#### 隐马尔科夫模型 Hidden Markov Model
潜在变量是服从多项式分布的离散变量$$z_n$$，描述了那个混合分量用于生成对应的观测$$x_n$$。与之前一样，比较方便的做法是使用1-of-K表示方法。我们现在让$$z_n$$的概率分布通过条件概率分布$$p(z_n|z_{n−1})$$对前一个潜在变量$$z_{n−1}$$产生依赖。由于潜在变量是$$K$$维二值变量，因此条件概率分布对应于数字组成的表格，记作$$A$$，它的元素被称为转移概率（transition probabilities）。元素为$$A_{jk}≡p(z_{nk}=1|z_{n−1,j}=1)$$。满足$$0≤A_{jk}≤1$$且$$\sum_k A_{jk} = 1$$，从而矩阵$$A$$有$$K(K−1)$$个独立的参数。这样，我们可以显式地将条件概率分布写成
<center>$$p(z_n|z_{n-1},A) = \prod\limits_{k=1}^K\prod\limits_{j=1}^KA_{jk}^{z_{n-1, j}z_{nk}} $$</center>
初始潜在结点$$z_1$$很特别，因为它没有父结点，因此它的边缘概率分布$$p(z_1)$$由一个概率向量$$π$$表示，元素为$$π_k≡p(z_{1k}=1)$$，即
<center>$$p(z_1 | \pi) = \prod\limits_{k=1}^K\pi_k^{z_{1k}}$$</center>
有时可以将状态画成状态转移图中的一个结点，这样就可以图形化地表示出转移矩阵。下图给出了K = 3的情形。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_6.png)
另一种表示方法将状态转移图在时间上展开，被称为晶格图（lattice diagram）或格子图（trellis diagram）。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_7.png)

可以通过定义观测变量的条件概率分布$$p(x_n|z_n,ψ)$$来确定一个概率模型，其中$$ψ$$是控制概率分布的参数集合，这些条件概率被称为发射概率(emission probabilities)。由于$$x_n$$是观测值，因此对于一个给定的$$ψ$$值，概率分布$$p(x_n|z_n,ψ)$$由一个$$K$$维的向量组成，对应于二值向量$$z_n$$的$$K$$个可能的状态。我们可以将发射概率表示为
<center>$$p(x_n|z_n,\psi) = \prod\limits_{k=1}^Kp(x_n|\phi_k)^{z_{nk}} $$</center>
我们将注意力集中在同质的（homogeneous）模型上，其中所有控制潜在变量的条件概率分布都共享相同的参数$$A$$，类似地所有发射概率分布都共享相同的参数$$ψ$$，从而观测变量和潜在变量上的联合概率分布为
<center>$$p(X,Z|\theta) = p(z_1|\pi)\left[\prod\limits_{n=2}^N p(z_n|z_{n−1},A)\right]\prod\limits_{m=1}^Np(x_m|z_m,\phi)$$</center>

从生成式的观点考虑隐马尔科夫模型，我们可以更好地理解隐马尔科夫模型。回忆一下，为了从一个混合高斯分布中生成样本，我们首先随机选择一个分量，选择的概率为混合系数$$π_k$$，然后从对应的高斯分量中生成一个样本向量$$x$$。这个过程重复$$N$$次，产生$$N$$个独立样本组成的数据集。在隐马尔科夫模型的情形，这个步骤修改如下：  
首先我们选择初始的潜在变量$$z_1$$，概率由参数$$π_k$$控制，然后采样对应的观测$$x_1$$。现在我们使用已经初始化的$$z_1$$的值，根据转移概率$$p(z_2|z_1)$$来选择变量$$z_2$$的状态。从而我们以概率$$A_{jk}$$选择$$z_2$$的状态$$k$$，其中$$k=1,...,K$$。一旦我们知道了$$z_2$$，我们就可以对$$x_2$$采样，从而也可以对下一个潜在变量$$z_3$$采样，以此类推。

下图说明了从隐马尔科夫模型生成样本的过程。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_8.png)
这个模型的潜在变量$$z$$有三个状态，发射概率$$p(x|z)$$是高斯概率，其中$$x$$是二维的。(a)发射概率密度为常数的轮廓线，对应于潜在变量的三个状态。(b)从隐马尔科夫模型中抽取的50个样本点，数据点的颜色对应于生成它们的分量的颜色，数据点之间的连线表示连续的观测。这里，转移矩阵是固定的。在任何状态，都有5%的概率转移到每个其他的状态，有90%的概率保持相同的状态。

这个标准的HMM模型有很多变体，例如可以通过对转移矩阵A的形式进行限制的方式进行限制。例如从左到右HMM（left-to-right HMM），它将$$A$$中$$k<j$$的元素$$A_{jk}$$设置为$$0$$。一旦离开了某个状态，就无法再次回到这个状态。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_9.png)
通常对于这种模型，初始状态概率p(z1)被修改，使得$$p(z_{11})=1$$。也就是说，每个序列被限制为从状态j=1开始。这种模型的晶格图如下。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_10.png)

隐马尔科夫模型的许多应用，例如语音识别或在线字符识别都使用了这种从左到右的结构。在线手写数字识别中每个手写数字由钢笔的轨迹与时间的函数表示，函数的形式是钢笔坐标的一个序列。与离线手写数字的例子不同，离线数据集由二维像素化的图像组成。
![](https://mqshen.gitbooks.io/prml/content/Chapter13/images/13_11.png)
第一行：在线手写数字的例子。第二行：生成式地采样得到的数字，模型时一个从左到右的隐马尔科夫模型，在45个手写数字组成的数据集上进行训练。

这里，我们在由45个数字“2”的例子组成的数据子集上训练一个马尔科夫模型。有$$K=16$$种状态，每个状态可以生成可以生成固定长度的线段，它具有16种可能的角度中的一个，因此发射概率是一个$$16×16$$的概率表，与每个状态下标的值所允许的角度值相关联。除了那些使得状态下标k不变或加1的转移概率之外，其他的转移概率全部被设置为0。

**隐马尔科夫模型的一个强大的性质是它对于时间轴上局部的变形（压缩和拉伸）具有某种程度的不变性。考虑在线手写数字例子中，数字“2”的书写方式。书写风格的自然的变化会使得这两个部分的相对大小发生变化。从生成式的观点来看，这种变化可以整合到隐马尔科夫模型中，方法是改变状态模型中保持在同一个状态的转移的数量和在连续的状态之间转移的数量。在语音识别的问题中，对时间轴的变形与语速的自然变化相关，隐马尔科夫模型可以适应这种变形，不会对这种变形赋予过多的惩罚。**
