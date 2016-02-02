---
layout:     post
title:      "图模型"
subtitle:   "贝叶斯网络，马尔科夫随机场，联合概率分解，条件独立表示，图的概率推断，条件随机场"
date:       2016-02-01 23:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---
##### 贝叶斯网络
根据有向图模型，可以写出对应的联合概率表达式。联合概率表达式由一系列条件概率的乘积组成，每一项对应于图中的一个结点。每个这样的条件概率分布只以图中对应结点的父结点为条件。因此，对于一个有K个结点的图，联合概率为
<center>$$p(x) = \prod\limits_{k=1}^Kp(x_k|pa_k) $$</center>
其中$$p_ak$$表示$$x_k$$的父结点的集合，$$x={x_1,...,x_K}$$。这个关键的方程表示有向图模型的联合概率分布的分解（factorization）属性。

我们考虑的有向图要满足一个重要的限制，即不能存在有向环（directed cycle）。这种没有有向环的图被称为有向无环图（directed acyclic graph），或DAG。

贝叶斯多项式拟合模型是有向图描述概率分布的一个例子。联合概率分布等于先验概率分布$$p(w)$$与$$N$$个条件概率分布$$p(t_n|w)$$的乘积：
<center>$$p(t,w) = p(w)\prod\limits_{n=1}^Np(t_n|w) $$</center>
图模型表示的联合概率分布如图所示：
![](https://mqshen.gitbooks.io/prml/content/Chapter8/bayesian/images/directed_graphical_probability.png)
这种图结构中，我们画出一个单一表示的结点$$t_n$$，然后用一个被称为板（plate）的方框圈起来，标记为$$N$$，表示有$$N$$个同类型的点。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/bayesian/images/directed_graphical_compact.png)
我们有时会发现，显式地写出模型的参数和随机变量是很有帮助的。
<center>$$p(t,w|x,\alpha,\sigma^2) = p(w|\alpha)\prod\limits_{n=1}^Np(t_n|w,x_n,\sigma^2)$$</center>
对应地，我们可以在图表示中显式地写出$$x$$,$$α$$。随机变量由空心圆表示，确定性参数由小的实心圆表示。另外，在图模型中，我们通过给对应的结点加上阴影的方式来表示观测变量（observed variables），如将变量$${t_n}$$根据多项式曲线拟合中的训练集进行设置。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/bayesian/images/directed_graphical_observed.png)
注意，$$w$$不是观测变量，因此$$w$$是潜在变量（latent variable）的一个例子。潜在变量也被称为隐含变量（hidden variable）。

通常，因为我们的最终目标是对输入变量进行预测，所以对$$w$$这样的参数本身不感兴趣。假设给定一个输入值$$x$$，我们想找到以观 测数据为条件的对应的$$t$$的概率分布。描述这个问题的图模型如图所示。
![多项式回归模型](https://mqshen.gitbooks.io/prml/content/Chapter8/bayesian/images/directed_graphical_regression.png)
以确定性参数为条件，这个模型的所有随机变量的联合分布为
<center>$$p(\hat{t},t,w|\hat{x},x,\alpha,\sigma^2) = \left[\prod\limits_{n=1}^Np(t_n|x_n,w,\sigma^2)\right]p(w|\alpha)p(\hat{t}|\hat{x},w,\sigma^2) $$</center>
然后，根据概率的加法规则，对模型参数w积分，得到t^的预测分布
<center>$$p(\hat{t}|\hat{x},x,t,\alpha,\sigma^2) \propto \int p(\hat{t},t,w|\hat{x},x,\alpha,\sigma^2)dw$$</center>
其中我们隐式地将$$t$$中的随机变量设置为数据集中观测到的具体值。

图模型描述了生成观测数据的一种因果关系（causal）过程。因此，这种模型通常被称为生成式模型（generative model）。多项式回归模型因为没有与输入变量$$x$$相关联的概率分布，所以不是生成式模型，因此无法从这个模型中人工生成数据点。通过引入合适的先验概率分布$$p(x)$$，我们可以将模型变为生成式模型，代价是增加了模型的复杂度。

概率模型中的隐含变量不必具有显式的物理含义。它的引入可以仅仅为了从更简单的成分中建立一个更复杂的联合概率分布。

两种情形很值得注意，即父结点和子结点都对应于离散变量的情形，以及它们都对应高斯变量的情形，因为在这两种情形中，关系可以层次化地推广，构建任意复杂的有向无环图。

##### 条件独立

如果一组变量的联合概率分布的表达式是根据条件概率分布的乘积表示的（即有向图的数学表达形式），那么原则上我们可以通过重复使用概率的加和规则和乘积规则测试是否具有潜在的条件独立性。在实际应用中，这种方法非常耗时。图模型的一个重要的优雅的特征是，联合概率分布的条件独立性可以直接从图中读出来，不用进行任何计算。完成这件事的一般框架被称为“d-划分”（d-separation），其中“d”表示“有向（directed）。

现在，开始讨论有向图的条件独立性质。考虑三个简单的例子：

![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/first_three.png)
<center>$$p(a,b,c) = p(a|c)p(b|c)p(c)$$</center>
如果没有变量是观测变量，那么我们可以通过对式两边进行积分或求和的方式，考察$$a,b$$是否是相互独立的，即
<center>$$p(a,b) = \sum\limits_c p(a|c)p(b|c)p(c)$$</center>
一般地，这不能分解为乘积$$p(a)p(b)$$，因此
<center>$$a \not\perp b | \varnothing  $$</center>
其中$$∅$$表示空集，符号$$\not\perp$$表示条件独立性质不总是成立。

现在假设我们以变量$$c$$为条件,我们可以很容易地写出给定$$c$$的条件下，$$a,b$$的条件概率分布，形式为
\begin{eqnarray}
p(a,b|c) &=& \frac{p(a,b,c)}{p(c)} \\
&=& p(a|c)p(b|c)
\end{eqnarray}
因此我们可以得到条件独立性质
<center>$$a \perp b | c$$</center>
结点$$c$$被称为关于这个路径“尾到尾”（tail-to-tail），因为结点与两个箭头的尾部相连。当我们以结点$$c$$为条件时，被用作条件的结点“阻隔”了从$$a$$到$$b$$的路径，使得$$a,b$$变得（条件）独立了。

我们可以类似地考虑下面给出的图。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/second_three.png)
这幅图对应的联合概率分布形式为
<center>$$p(a,b,c) = p(a)p(c|a)p(b|c)$$</center>
首先，假设所有的变量都不是观测变量。与之前一样，我们可以考察$$a,b$$是否是相互独立的，方法是对$$c$$积分或求和，得到
<center>$$p(a,b) = p(a)\sum\limits_cp(c|a)p(b|c) = p(a)p(b|a)$$</center>
这个结果与之前的相同。现在假设我们以结点$$c$$为条件，如图所示。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/second_three_conditional.png)
使用贝叶斯定理
\begin{eqnarray}
p(a,b|c) &=& \frac{p(a,b,c)}{p(c)} \\
&=& \frac{p(a)p(c|a)p(b|c)}{p(c)} \\
&=& p(a|c)p(b|c)
\end{eqnarray}
从而我们又一次得到了条件独立性质
<center>$$a \perp b | c$$</center>
结点c被称为关于从结点$$a$$到结点$$b$$的路径“头到尾”(head-to-tail)。

最后，我们考虑第三个例子，如图所示。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/third_three.png)
联合概率分布:
<center>$$p(a,b,c) = p(a)p(b)p(c|a,b)$$</center>
首先考虑当没有变量是观测变量时的情形。两侧关于$$c$$积分或求和，得到
<center>$$p(a,b) = p(a)p(b)$$</center>
因此当没有变量被观测时，$$a,b$$是独立的，这与前两个例子相反。我们可以把这个结果写成
<center>$$a \perp b | \varnothing $$</center>
现在假设我们以$$c$$为条件
![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/third_three_conditional.png)
但是以结点$$c$$的值为条件。这张图中，引入条件结点使得$$a,b$$之间产生了依赖关系。
因此，我们第三个例子与前两个例子的行为相反。图形上，因为c连接了两个箭头的头，所以我们说结点c关于从a到b的路径是“头到头”（head-to-head）。当结点$$c$$没有被观测到的时候，它“阻隔”了路径，从而变量$$a,b$$是独立的。然而，以$$c$$为条件时，路径被“解除阻隔”，使得$$a,b$$相互依赖了。

如果存在从结点$$x$$到结点$$y$$的一条路径，其中路径的每一步都沿着箭头的方向，那么我们说结点$$y$$是结点$$x$$的后继（descendant）。这样，可以证明，在一个头到头的路径中，如果任意结点或者它的任意一个后继被观测到，那么路径会被“解除阻隔”。

总之，一个尾到尾结点或者头到尾结点使得一条路径没有阻隔，除非它被观测到，之后它就阻隔了路径。相反，一个头到头结点如果没有被观测到，那么它阻隔了路径，但是一旦这个结点或者至少一个后继被观测到，那么路径就被“解除阻隔”了。

###### D-划分

考虑从A中任意结点到B中任意结点的所有可能的路径。我们说这样的路径被“阻隔”，如果它包含一个结点满足下面两个性质中的任何一个:

1. 路径上的箭头以头到尾或者尾到尾的方式交汇于这个结点，且这个结点在集合C中。
2. 箭头以头到头的方式交汇于这个结点，且这个结点和它的所有后继都不在集合C中。

如果所有的路径都被“阻隔”，那么我们说$$C$$把$$A$$从$$B$$中D-划分开，且图中所有变量上的联合概率分布将会满足$$A⊥B|C$$。

![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/d_separation.png)
在左图中，从$$a$$到$$b$$的路径没有被结点$$f$$阻隔，因为对于这个路径来说，它是一个尾到尾结点，并且没有被观测到。这条路径也没有被结点$$e$$阻隔，因为虽然后者是一个头到头的结点，但是它有一个后继$$c$$在条件集合中，且被观测到。因此条件独立关系$$a⊥b|c$$在这个图中不成立。在右图中，从$$a$$到$$b$$的路径被结点$$f$$阻隔，因为它是一个尾到尾的结点，并且被观测到，因此使用这幅图进行分解的任何概率分布都满足条件独立性质$$a⊥b|f$$。注意，这个路径也被结点$$e$$阻隔，因为$$e$$是一个头到头的结点，并且它和它的后继都没在条件集合中。

###### 马尔科夫毯 Markov Blanket
考虑一个联合概率分布$$p(x_1,...,x_D)$$，它由一个具有$$D$$个结点的有向图表示。考虑变量$$x_i$$对应的结点上的条件概率分布，其中条件为所有剩余的变量$$x_{j≠i}$$使用分解性质，我们可以将条件概率分布表示为下面的形式
\begin{eqnarray}
p(x_i|x_{\{j \neq i\}} &=& \frac{p(x_1,...,x_D)}{\int p(x_1,...,x_D)dx_i} \\
&=& \frac{\prod\limits_kp(x_k|pa_k)}{\int \prod\limits_kp(x_k|pa_k)dx_i}
\end{eqnarray}
我们现在观察到任何与$$x_i$$没有函数依赖关系的因子都可以提到$$x_i$$的积分外面，从而在分子和分母之间消去。唯一剩余的因子是结点$$x_i$$本身的条件概率分布$$p(x_i|pa_i)$$，以及满足下面性质的结点$$x_k$$的条件概率分布：结点$$x_i$$在$$p(x_k|pa_k)$$的条件集合中，即$$x_i$$是$$x_k$$的父结点。条件概率分布$$p(x_i|pa_i)$$依赖于结点$$x_i$$的父结点，而条件概率分布$$p(x_k|pa_k)$$依赖于$$x_i$$的子结点以及同父结点（co-parents），即那些对应于$$x_k$$（而不是$$x_i$$）的父结点的变量。由父结点、子结点、同父结点组成的结点集合被称为马尔科夫毯，如图所示。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/conditional/images/markov_blanket.png)
它的性质为：以图中所有剩余结点为条件，$$x_i$$的条件概率分布值依赖于马尔科夫毯中的变量。

##### 马尔科夫随机场 Markov random field
一个马尔科夫随机场（Markov random field），也被称为马尔科夫网络（Markov network）或无向图模型（undirected graphical model），包含一组结点，每个结点都对应着一个变量或一组变量。链接是无向的，即不含有箭头。在无向图的情形中，首先讨论条件独立性质是比较方便的。

在有向图的情形下，我们看到可以通过使用被称为D-划分的图检测方法判断一个特定的条件独立性质是否成立。这涉及到判断链接两个结点集合的路径是否被“阻隔”。马尔科夫随机场是另一种概率分布的图语义表示，使得条件独立性由单一的图划分确定。

。。。。待续

##### 树
在无向图的情形中，树被定义为满足下面性质的图：任意一对结点之间有且只有一条路径。于是这样的图没有环。在有向图的情形中，树的定义为：有一个没有父结点的结点，被称为根（root），其他所有的结点都有一个父结点。如果有向图中存在具有多个父结点的结点，但是在任意两个结点之间仍然只有一条路径（忽略箭头方向），那么这个图被称为多树（polytree）。

##### 因子图 factor graph
有向图和无向图都使得若干个变量的一个全局函数能够表示为这些变量的子集上的因子的乘积，因子图统一地显式地表示出了这个分解，方法是：在表示变量的结点的基础上，引入额外的结点表示因子本身。
<center>$$p(x) = \prod\limits_sf_s(x_s) $$</center>
在因子图中，概率分布中的每个变量都有一个结点（同样用圆圈表示），这与有向图和无向图的情形相同。还存在其他的结点（用小正方形表示），表示联合概率分布中的每个因子$$f_s(x_s)$$。最后，在每个因子结点和因子所依赖的变量结点之间，存在无向链接。由于因子图由两类不同的结点组成，且所有的链接都位于两类不同的结点之间，因此因子图被称为二分的（bipartite）。

同一个有向图或者无向图都可能对应于多个因子图，这使得因子图对于分解的精确形式的表示更加具体。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/inference/images/full_connect.png)
![](https://mqshen.gitbooks.io/prml/content/Chapter8/inference/images/directed_factor_graph.png)

##### 加和-乘积算法 sum-product algorithm
有向无环图的精确推断的置信传播（belief propagation）的算法等价于加-乘算法的一个具体情形。

我们假设模型中所有的变量都是离散的，因此求边缘概率对应于求和的过程。再假设原始的图是一个无向树或者有向树或者多树，从而对应的因子图有一个树结构。

我们可以将原始的图转化为因子图，使得我们可以使用同样的框架处理有向模型和无向模型。

我们的目标是利用图的结构完成两件事：(1)得到一个高效的精确推断算法来寻找边缘概率，(2)在需要求解多个边缘概率的情形，计算可以高效地共享。

首先，因子图用因子结点的乘积表示联合分布。
<center>$$p(x) = \prod\limits_{s }f_s(\mathbb{x_s})$$</center>
对于特定的变量结点$$x$$，我们寻找边缘概率$$p(x)$$。现阶段，我们假设所有的变量都是隐含变量。根据定义，边缘概率分布通过对所有$$x$$之外的变量上的联合概率分布进行求和的方式得到，即
<center>$$p(x) = \sum\limits_{\mathbb{x}\\x}p(\mathbb{x})$$</center>
算法的思想是使用因子图的因子乘积表达式替换$$p(x)$$，然后交换加和与乘积的顺序，得到一个高效的算法。
![](http://i13.tietuku.com/8d89c9673305450b.png)
如图的联合概率分布可以写成乘积的形式
<center>$$p(x) = \prod\limits_{s \in ne(x)}F_s(x,X_s)$$</center>
其中$$ne(x)$$表示与$$x$$相邻的因子结点的集合，$$X_s$$表示子树中通过因子结点$$f_s$$与变量结点$$x$$相连的所有变量的集合，$$F_s(x,X_s)$$表示分组中与因子$$f_s$$相关联的所有因子的乘积。
交换加和与乘积的顺序，我们有

$$
\begin{eqnarray}
p(x) = \prod\limits_{s \in ne(x)}\left[\sum\limits_{X_s}F_s(x,X_s)\right] \\
= \prod\limits_{s \in ne(x)}\mu_{f_s \to x}(x)
\end{eqnarray}
$$

函数$$μ_{f_s}→x(x)$$可以被看做从因子结点$$f_s$$到变量结点$$x$$的信息（message）。我们看到，需要求解的边缘概率分布$$p(x)$$等于所有到达结点$$x$$的输入信息的乘积。

我们注意到每个因子$$F_s(x,X_s)$$由一个因子（子）图，因此本身可以被分解。

算法思想总结：我们可以将结点x看成树的根结点，x的边缘概率分布等于沿着所有到达这个结点的链接的输入信息的乘积。从叶结点开始计算，如果一个叶结点是一个变量结点，那么它沿着与它唯一相连的链接发送的信息为$$1$$。如果叶结点是一个因子结点，发送的信息的形式为$$f(x)$$。每个结点都可以向根结点发送信息。一旦结点收到了所有其他相邻结点的信息，那么它就可以向根结点发送信息。递归地传递信息，直到信息被沿着每一个链接传递完毕，并且根结点收到了所有相邻结点的信息。
![](https://mqshen.gitbooks.io/prml/content/Chapter8/inference/images/sum_product.png)
对于标准化系数的问题。如果因子图是从有向图推导的，那么联合概率分布已经正确的被标准化了，因此通过加-乘算法得到的边缘概率分布会类似的被正确标准化。然而，如果我们开始于一个无向图，那么通常会存在一个未知的标准化系数1/Z。

现在，考虑一个简单的例子：
![](https://mqshen.gitbooks.io/prml/content/Chapter8/inference/images/factor_sum_product.png)
它的未标准化联合概率分布为
<center>$$\tilde{p}(x) = f_a(x_1,x_2)f_b(x_2,x_3)f_c(x_2,x_4)$$</center>
让我们令结点$$x_3$$为根结点，此时有两个叶结点$$x_1$$,$$x_4$$。从叶结点开始，我们有
<center>$$
\begin{eqnarray}
\mu_{x_1 \to f_a}(x_1) = 1  \\
\mu_{f_a \to x_2}(x_2) = \sum\limits_{x_1}f_a(x_1, x_2)  \\
\mu_{x_4 \to f_c}(x_4) = 1 \\
\mu_{f_c \to x_2}(x_2) = \sum\limits_{x_4}f_c(x_2, x_4) \\
\mu_{x_2 \to f_b}(x_2) = \mu_{f_a \to x_2}(x_2)\mu_{f_c \to x_2}(x_2)  \\
\mu_{f_b \to x_3}(x_3) = \sum\limits_{x_2}f_b(x_2, x_3)\mu_{x_2 \to f_b}(x_2)
\end{eqnarray}
$$</center>
一旦信息传播完成，我们就可以将信息从根结点传递到叶结点，这些信息为
<center>$$
\begin{eqnarray}
\mu_{x_3 \to f_b}(x_3) = 1  \\
\mu_{f_b \to x_2}(x_2) = \sum\limits_{x_3}f_b(x_2, x_3) \\
\mu_{x_2 \to f_a}(x_2) = \mu_{f_b \to x_2}(x_2)\mu_{f_c \to x_2}(x_2) \\
\mu_{f_a \to x_1}(x_1) = \sum\limits_{x_2}f_a(x_1, x_2)\mu_{x_2 \to f_a}(x_2) \\
\mu_{x_2 \to f_c}(x_2) = \mu_{f_a \to x_2}(x_2)\mu_{f_b \to x_2}(x_2)  \\
\mu_{f_c \to x_4}(x_4) = \sum\limits_{x_2}f_c(x_2, x_4)\mu_{x_2 \to f_c}(x_2)
\end{eqnarray}
$$</center>
现在一个信息已经在两个方向上通过了每个链接，因此我们现在可以计算边缘概率分布。作为一个简单的检验，让我们验证边缘概率分布$$p(x_2)$$由正确的表达式给出。使用上面的结果将信息替换掉，我们有

$$
\begin{eqnarray}
\tilde{p}(x_2) = \mu_{f_a \to x_2}(x_2)\mu_{f_b \to x_2}(x_2)\mu_{f_c \to x_2}(x_2) \\
= \left[\sum\limits_{x_1}f_a(x_1,x_2)\right]\left[\sum\limits_{x_3}f_b(x_2,x_3)\right]\left[\sum\limits_{x_4}f_c(x_2,x_4)\right] \\
= \sum\limits_{x_1}\sum\limits_{x_3}\sum\limits_{x_4}f_a(x_1,x_2)f_b(x_2,x_3)f_c(x_2,x_4) \\
= \sum\limits_{x_1}\sum\limits_{x_3}\sum\limits_{x_4}\tilde{p}(x)
\end{eqnarray}
$$

##### 最大和算法 max-sum
加乘算法能够将联合概率分布表示为一个因子图，并且高效地求出成分变量上的边缘概率分布。但是怎么找到联合概率分布的最大值以及相应的变量设置呢？

一个想法是用加乘算法找到每个点边缘概率分布，分别求出使得边缘概率最大的变量值。然而，这一组值每个值都单独取得最大的概率，并不能使得联合概率分布具有最大值。

在推导加-乘算法时，我们使用了乘法的分配律。最大化也可以类似进行：
<center>$$\max(ab,ac) = a \max(b,c)  $$</center>
这对于$$a≥0$$的情形成立（这对于图模型的因子总成立）。这使得我们交换乘积与最大化的顺序。

$$
\begin{eqnarray}
\max_x p(x) = \frac{1}{Z}\max_{x_1}\dots\max_{x_M}[\psi_{1,2}(x_1,x_2)\dots\psi_{N-1,N}(x_{N-1},x_N)] \\
=  \frac{1}{Z}\max_{x_1}\left[\max_{x_2}\left[\psi_{1,2}(x_1,x_2)\left[\dots\max_{x_N}\psi_{N-1,N}(x_{N-1},x_N)\right]\dots\right]\right]
\end{eqnarray}
$$
正如边缘概率的计算一样，我们看到交换最大值算符和乘积算法会产生一个更高效的计算，且更容易表示为从结点xN沿着结点链传递回结点$$x_1$$的信息。最后对所有到达根结点的信息的乘积进行最大化，这可以被称为最大化乘积算法（max-produce algorithm），与加-乘算法完全相同唯一的区别是求和被替换为了求最大值。注意，现阶段信息被从叶结点发送到根结点，而没有相反的方向。

在实际应用中，许多小概率的乘积可以产生数值下溢的问题，因此更方便的做法是对联合概率分布的对数进行操作。对数函数是一个单调函数，因此求最大值的运算符可以与取对数的运算交换顺序，因此我们得到了最大化和算法（max-sum algorithm）。

如何寻找联合概率达到最大值的变量的配置呢？找到根节点变量的概率最大值后，将信息从根结点传回叶结点，然而，由于我们现在进行的是最大化过程而不是求和过程，因此有可能存在多个$$x$$的配置，它们都会给出$$p(x)$$的最大值。我们需要确定对应于同样的最大化配置的前一个变量的状态。可以通过跟踪变量的哪个值产生了每个变量的最大值状态，即存储下面的量
<center>$$\phi(x_n) =  \arg\max_{x_{n-1}}\left[\ln f_{n-1,n}(x_{n-1},x_n) + \mu_{x_{n-1} \to f_{n-1, n}(x_{n-1})}\right] $$</center>
对于给定变量的每个状态，存在前一个变量的一个唯一的状态使得概率取最大值。一旦我们知道了最终结点xN的最可能的值，我们就可以沿着链接回退，找到结点$$x_{N−1}$$的最可能状态，并且以此类推，回到最初的结点$$x_1$$。这对应于将信息沿着链进行反方向的传递，使用下面的公式
<center>$$x_{n-1}^{max} = \phi(x_n^{max})$$</center>
被称为反向跟踪（back-tracking）。注意，可能存在多个xn−1的值，每个都能给出最大值。在进行反向跟踪时，只要我们选择了这些变量中的一个，那么我们就能够保证得到一个全局相容的最大化配置。

##### 条件随机场 Conditional Random Fields
条件随机场是一种判别式（discriminative）概率无向图模型，被用来对连续随机变量的特征进行分类，用于标注和切分有序数据的条件概率。

如果给定的马尔可夫随机场MRF中每个随机变量下面还有观察值，我们要确定的是给定观察集合下，这个MRF的分布，也就是条件分布，那么这个MRF就称为CRF。它的条件分布形式完全类似于MRF的分布形式，只不过多了一个观察集合x。因此我们可以认为CRF本质上是给定了观察值(observations)集合的MRF。

可以推导出CRF表示条件概率的公式：

用训练数据计算出使得后验概率最大的参数值，我们采用最小化负对数后验，再转化为似然乘以先验，先验可以用高斯估计。但是似然概率并不可解，我们可以用伪似然pseudo-likelihood去近似。只计算马尔可夫毯Markov blanket。
