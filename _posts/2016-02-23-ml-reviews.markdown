---
layout:     post
title:      "Machine Learning Exam Review"
subtitle:   ""
date:       2016-02-23 14:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---

##### Name and describe the categories of learning. Name two concrete examples for learning algorithms.

- Unsupervised Learning: clustering, density estimation
- Supervised Learning: Regression, SVM, HMM, Boosting
- discriminant function (no prob. Learn a function), discriminant model(estimate the posterior for each class), and generative model (likelihoods and Bayes rule for prosterior.)
- Reinforcement Learning: no supervision, but a reward function.
- Generative model: kNN: given data points, assign each new data to its K nearest neighbors in feature space. Compute posteriors and MAP.

##### What is the purpose of regression and what is the general principle?

- Given a set of objects and a set of object categories (classes), regression learns a mapping function, then it can predict the categories of new object.

- General principle: first, extract the features of the input data using basis functions, then choose a regression model to take the features as input and compute the output. Next we compare the output of the model and desired output to evaluate the regression model, for example, use sum of square errors. and revise the model to fit out training data. But we need to avoid the overfitting.

##### Explain linear regression and what exactly is linear there.

The basis functions are nonlinear with respect to the input variables. But linear regression models are linear combination of nonlinear basis functions, so the model is linear to the parameters of the basis functions.

To evaluate the function, we need an error function, for example the sum of squared errors. Then we want to find the optimal parameters to minimize the error function by deriving the function w.r.t parameters. After obtaining the parameters, we can use the model to predict the target value of new input data.

##### What is a pseudoinverse matrix?

Pseudoinverse matrix is a generalization matrix inverse to nonsquare matrices. Every matrix has a pseudoinverse matrix, but only when the matrix is square and invertable, the pseudoinverse matrix equals to the inverse of the matrix.
Φ† ≡ (ΦTΦ) −1 ΦT
It can be computed using the singular value decomposition.

##### Name two types of basis functions and explain how they should be used.

- Gaussian basis function:

- Sigmoidal basis function: using logistic sigmoid function / ‘tanh’ function, the μj govern the locations of the basis functions in input space, and the parameter s governs their spatial scale. It is no need to normalize these basis function.

##### Explain regularization in the context of regression. Why is this useful?

After training the model on the training dataset, there will be the overfitting problem when to predict the new data. For example, the polynomials coefficients will become very large. The model gains the memory ability but decreases the ability of generalization. We can add a penalty term of the coefficients to the error function to avoid overfitting. This technique is called regularization.

##### What is the probabilistic formulation of regression? What is the equivalent of maximum-likelihood and maximum-a-posteriori?

We can assume that y is affected by Gaussian noise, the we can the likelihood of the measured data given a model. We want to find the model parameters to maximize the prob.
Maximum likelihood under Gaussian noise is equivalent to sum of square errors regression.
Maximum-a-posteriori assuming a Gaussian prior is equivalent to the regularized regression.
Sequential data: use the old data to calculate the posteriori as the prior of the new data to iteratively compute the parameters by MLE.
Predictive model: use the old data to calculate the posteriori as the prior of the new data to predict by integrating over all possible parameters.

##### Name the different kinds of Probabilistic Graphical Models and draw two examples. What do they represent?

Bayesian Network: directed graphical model
 p(a, b, c) = p(c | a, b) p(b | a) p(a).
absence of edges  independence.  One-to-one mapping, causal relationships
Markov Random Fields: undirected graphical models
 A ⊥ B | C.

##### What is a perfect map between a graphical model and a joint probability distribution?

I-map: Every D-separation  a conditional independence (fully connected)
D-Map: every conditional independence  D-separation
Perfect-Map: one-to-one mapping
every conditional independence property of the distribution is reflected in the graph, and vice versa, then the graph is said to be a perfect map for that distribution
What is D-separation?

(independence does not imply conditional independence)
Conditional independence properties of the joint distribution can be read directly from the directed acyclic graph by using D-separation.
There are three cases:
(a) the arrows on the path meet either head-to-tail or tail-to-tail at the node, and the node is in the set C, or
(b) the arrows meet head-to-head at the node, and neither the node, nor any of its descendants, is in the set C.
If all paths are blocked, then A is said to be d-separated from B by C, and the joint distribution over all of the variables in the graph will satisfy A ⊥ B | C.

##### What is the definition of the Markov blanket and how can it be computed in directed and undirected graphical models?

Markov blanket is the minimal set of observed nodes to obtain conditional independence.
For an undirected graph, it is dependent only on the neighbouring nodes.
For directed graph, the Markov blanket comprises the set of parents, children and co-parents of the node.


##### What is a Markov Random Field?

Markov Random Field is an undirected graph model to check conditional independence. Its node represents a variables or group of variables. It is defined as a factorization over clique potentials and normalized globally.

Two nodes are not connected in MRF  conditional independent given all other nodes.
Clique: fully connected subgraph.
Maximal clique: can not be extended without loosing fully connectivity.


##### How does the expressiveness of directed and undirected graphical models compare?

Directed graphs are useful for expressing causal relationships between random variables, whereas undirected graphs are better suited to expressing soft constraints between random variables without ordering, for example, the pixels in a camera image. Undirected graph is simpler and more intuitive to express conditional independence.

##### What is “explaining away” in the context of graphical models?

For head-to-head case, observations of child nodes will not block paths to the co-parents. We must observe the co-parent nodes also.

##### How to convert directed graphs to undirected graphs?
Conditional distributions in the directed graph are mapped to cliques in the undirected graph and connect all parents of head-to-head nodes.
Problem: remove conditional independence relations. There is no one-to-one mapping.

##### What is the general idea of the inference on an undirected Markov Chain?

When computing the local marginal distribution using the joint probability, we can rearrange the summation order of the production. Then we can recursively compute and store the all forward and backward messages. Next we can compute the normalizing factor for any node to get the marginal for this node.
It is a special case of sum-product algorithm.

##### What is a factor graph?

A generalization of directed and undirected graph by introducing additional nodes for the factors. It allows more explicit details of factorization.

Max-sum algorithm: to find the setting that maximize the joint probability and the maximal value. Max operator is distributive so we can exchange the order of it with production. Two problems: 1. Lots multiplication  log space. 2. Multiple configurations  keep track of them
CRF: Conditional Random Fields
For classification, discriminative model
Training: MAP, approximate likelihood using Markov blanket (Pseudo likelihood)
Potential function: positive, log-linear

##### What does the sum-product algorithm compute and how?

Sum-product algorithm computes the marginal distribution at a given node.
1.	It considers the node as a root node.
2.	Then initialize leaf node either 1 if variable or f(x) if factor.
3.	Propagate the message from the leaves to the root
4.	Then propagate the message from the root to the leaves
5.	Then we can get the marginal at every node by multiplying all incoming messages.

##### What is the definition of a Hidden Markov Model?

Sequential data, stationary = same generative distribution = i.i.d
By the introduction of discrete latent variables  HMM
Assume that it is the latent variables that form a Markov chain
It contains 1. discrete observation variables and state variables each with K states. 2. Transition model, first order Markov chain. 3. Observation model.

application: 1. Likelihood 2. Filtering and smoothing given observations 3.  Best sequence given observation 4. Optimal model parameters.
Filtering: forward algorithm
Smoothing: forward and backward algorithm
Most likely states: Viterbi
Optimal parameters: Baum-Welch or EM

##### How does the Viterbi algorithm work?  
It is same with max-sum algorithm if treat HMM as a directed tree.
From 1 to N time step, we assume at time step t, state j is the most probable path, denoted as μ_t(j). It can be compute from step t-1, μ_t-1(i) multiply transition probability and emission probability. At the same time, we compute the argmax of i.
On termination, we can get the maximal value of joint probability and we can backtrack the state setting for this sequence.

Baum-Welsh (EM): use k-means to initialize parameters. In E-step to compute the Q, and M-step to maximize Q. in each iteration, forward-backward algorithm is used. Result gives a local optimum.

##### How does k-means clustering work?
Given data set and number of cluster K. k-means can get the optimal assignments of each point and the cluster centers by minimizing the sum of square errors. It iteratively computes center μ and the point assignments r. At first, it initializes fixes the centers and find the optimal assignments. Then using these assignments to update the cluster centers. Repeat until convergence.

##### Name the definition of a Gaussian Mixture Model. What are the individual components and where can it be used?

Assume that data consists of K clusters and each cluster is Gaussian. And each data point is softly assigned into a cluster with a probability and a latent variable to indicate whether it is in a cluster or not.
Each individual component is a Gaussian.
It can be used for clustering in speech recognition.

##### What are the differences and similarities between k-means and Expectation Maximization in Gaussian Mixture Models?

k-means is a special case of GMM. They both contains expectation step and maximization step. And they both only find the local maximum.
But in k-means, the points are assigned to only one cluster. It is hard assignment. While in GMM, the points are assigned with probabilities. It is soft assignment.
In GMM, the covariance is also estimated by EM.
The number of iterations of EM in GMM is much higher and it needs more computation. So k-means can be used to initialize GMM.

##### What is the complete-data log-likelihood and how is it used in EM?

We assume that we can observe the latent variables Z, then we call the data set of X and Z is complete. The complete-date log-likelihood is about the joint distribution of X and Z.
Advantage is that the log-likelihood is much simpler, the log is inside the sum and rearrange the sums to reduce computation.
But in EM, we don’t know about the latent variables. So we consider the expectation of joint log-likelihood under the latent variable distribution. And the expectation of latent variables for one point is the responsibilities of that point. Then we can replace the expectation of the latent variables with their responsibilities. Finally we can use EM to maximizes the expected complete date log-likelihood.

##### What exactly is optimized in EM and how?

We want to maximize the log-likelihood but it is very difficult, the sum is in log. But we can introduce the latent variables and consider the marginalizing over the latent variables of joint distribution of x and latent variable. The log form is much simpler.
Then iteratively find the optimal parameters.
For E-step, it only increases the lower bound. We try to find the maximal lower bound of the functional of proposal distribution q based on the old parameters to get a new q and the same time to minimize the KL-divergence. Q equals to posterior of Z.
In M-step, fix the proposal distribution Q and update the parameters. It makes KL again larger than zero and the log-likelihood is also maximized.
Name examples for applications of the EM algorithm.

It can be used to find the parameters of HMM.

##### Name 3 different activation functions used in Neural Networks. What is the advantage of ReLU over sigmoids?

1.	Threshold activation, it is a binary classifier
2.	Sigmoid activation, it is used to model probability and its gradient is less than 1.
3.	Rectified Linear Unit activation, max(0,phi), it can be used to model positive real numbers and its gradient is either 0 or 1. It can reduce sparsity and doesn’t have the vanishing problem when propagate the gradient. It is faster and more efficient.

##### What is the name of the training procedure in Neural Networks? How does it work?

Back Propagation:
Forward the input to generate the output and propagate the errors backwards to evaluate the derivatives of the error function with respect to the weights. The derivatives are used to compute the adjustments using gradient descent.

##### Name the advantages of CNNs over fully connected NNs.
It is sparse and weight sharing, so it is easier to train and have many fewer parameters than fully connected networks with the same number of hidden units.

It is invariant to certain transformation of the inputs. It is suitable for 2D structure, like image and speech.
Fully-connected does not take into account the spatial structure of the images.
Sketch a perceptron for binary classification with n inputs.



##### What is the principle of the AdaBoost algorithm?

Use a weak classifier and turn it into a strong one with arbitrarily small training error.
Sequentially fit classifiers by minimizing the exponential loss and compute the weights of classifiers and update the weights of wrong data points.
Name the definition of a weak and a strong classifier. Give examples for each.

Weak:
Decision stumps: axis-aligned hyper-plane that minimize the class error
Decision trees: decision is made at every node.
Strong: Adaboost

##### What other variants of AdaBoost exist?

LogitBoost uses logistic function and weighted least-squares regression.
GentleBoost
Gradient Boost uses the general form of loss function. In each iteration, it calculates the gradient w.r.t classifier function and try to find a basis function that is close to the gradient residual. Then add it to the former classifier.

##### How can AdaBoost be used for face detection?

Extract the features like haar-like features
Training the adaboost using the weak classifers. The weak classifier contains a threshold to tell the belief of a feature.
Because the scale of the image is different. So we gradually increase the scale in each loop.

##### What is a kernel function?

Kernel function maps two arguments into a real number to measure their similarity without any feature.
Name the Theorem of Mercer and explain the kernel trick.

If a kernel is symmetric and positive definite, there exists a mapping into a feature space so that the kernel represents an inner product in this higher dimensional space.
Kernel trick is to express similarities of data points in term of an inner product and replace all inner products by the kernel function
It means that we don’t need to find the specific basis functions. We only need to find some kernel functions corresponding to some basis functions in feature space.

##### How can kernelization be applied to k-means clustering?

We can replace each point as basis function, and rewrite the center also.
The square error can be rewrited with kernels.

##### What is the principle of kernel PCA?

PCA: project data onto a subspace of lower dimension so that the variance is maximized, decorrelation.
Reconstruction: minimize the squared error. Substract mean, project into eigenvectors and back project and add mean.
For data is non-linearly distributed along the principle component, we use non-linear kernel to map into a space we can do PCA.
Construct the inner products of the basis functions and replace the basis functions by kernels.

##### What is a Support Vector Machine?

SVM is a binary classifier that learn a linear discriminant function which is a decision hyper-plane that maximize the margin to separate the data points.

Support vectors has minimal distance to decision plane and the distance of support vectors is rescaled to 1.
Use Lagrange constrained optimization to derive the dual formulation and then replace with kernel to get the optimal solutions. The trick is we can replace the kernel by any other valid kernel and obtain again an optimal solution.
For prediction process, we only need to compute new data point with support vectors.
For multiple classes, one-to-many classification.
For non-separable problem, we allow data points that located within margins with slack variables to classify them as correct classified ones or misclassified ones.  

##### Name the definition of a Gaussian process. What specifies a GP? Draw an example.
A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is a Gaussian distribution over functions.
To specify a GP: mean function and covariance function.
How to handle infinity? Split into a finite part and an infinite part and marginalize the infinite part.

##### How can a function be sampled from a Gaussian Process?

1.	Choose some input points
2.	Compute the covariance matrix K of the input points using the kernel function
3.	Generate a random	Gaussian vector from the GP, zero mean, K covariance
4.	Plot the x and y.


##### What are the hyperparameters of the squared exponential kernel?

Many names: squared exponential, radial basis function, Gaussian kernel
Signal variance
Length scale: range between points, influence distance
Noise variance

##### What is Automatic Relevance Determination?

During the optimization process to learn the hyper-parameters, the reciprocal length scale in the squared exponential kernel can be interpreted as the weight for each feature dimension. So during the training process, the relevance is automatically determined.

##### What is the main problem in Gaussian Process classification in contrast to regression and how can it be solved?

When computing the predictive distribution, we marginalize over the latent functions from the training data. If using the Bayes rules to transform the posterior of the latent functions, the likelihood is a sigmoid function not a Gaussian. We can not compute in closed form.
We only can approximate the posterior. We could use Laplace approximation, expectation propagation or Variational methods.
Laplace approximation is based on the second-order Taylor expansion to find the mean and variance of the Gaussian.

##### Name and describe briefly 4 different sampling methods.

1.	Probability transformation:
a.	inverse of cumulative distribution function
need to calculate and inverse.
2.	Rejection sampling:
a.	Find a proposal distribution which is easy to sample.
b.	Find a number k, let kq>=p. p is target distribution that is under kq.
c.	Sample from x-axis uniformly get a x0.
d.	Sample from 0 to kq(x0).
e.	Accept or reject.
Inefficient. Might always reject in high dimensional space.
3.	Importance sampling:
a.	Assign an importance weight to each sample from q.
b.	The probability of certain sample is in the interval A is the area under the curve p.
c.	This equals to the expectation of the indicator function under the curve p.
d.	The trick to replace this expectation under the distribution q.
e.	Then use number of samples with weight to approximate.
4.	MCMC:
a.	Main idea: sample from a proposal distribution and design its transition matrix results stationary distribution to be our desired distribution.
b.	Theorem: a Markov chain with transition matrix A is irreducible and satisfies the detailed balance w.r.t pi (chain is reversible)  pi is the stationary distribution. (easy proof pi = pi * A)
c.	MH algorithm:
i.	For each state:
1.	Sample x’from q(x’|x)
2.	Compute acceptance probability
3.	Compute r=min(1,a)
4.	Sample U[0,1]
5.	Compare, less to keep and set it as next state

##### How can a Particle Filter be used for Markov Localization?

1.	Each particle is a hypothesis of position
2.	Proposal distribution is the motion model to predict
3.	The observation model is used to compute the importance weight to correct the prediction.
4.	For every sample, each loop sample a new one and compute the weights and add into the set.
5.	Resampling each sample with probability proportional to weights.
Particle filter is non-parametric implementation of Bayes filter. By a set of random state samples to represent a probability distribution  can be not Gaussian  Can model non-linear
Basic principles: set of state hypothesis, survival of the fittest.
Resample: standard n times vs. low variance sampling: only samples once
Kidnapped Robot: add new uniform samples.
What is a stationary distribution in a Markov Chain?
Each step in the chain, distribution of states is invariant.
Existence: solve the equation: the eigenvector with eigenvalue 1.  every row has such eigenvector, but may not be unique.
Uniqueness: ergodic, A  irreducible: we can reach every state from any other state in finite step with non-zero probability.

##### How does the Metropolis-Hastings algorithm work?

Main idea: construct the transition matrix and make the resulting stationary distribution is our target distribution.
For each state:
	Sample x’ from q(x’|x);
		Compute the acceptance probabililty
		R = min(1, a)
		Sample from U(0,1)
		Accept as next state or reject

##### What is Gibbs Sampling?

Idea: for multiple variables, in each time step, sample each variable from the full conditional and replace the old one with this new sample. Then cycling through all variable.
It is a special case of MCMC algorithm when acceptance probability equals to 1. So it can be applied for high dimensional cases.

##### What is variational inference?

Variational inference takes functions as input and find functions to optimize a given functional. It is mainly used to find approximations to a given function.
Functional: a mapping that takes a set of functions as the input and that returns the real number of the functional as the output. E.g entropy.

##### Name the definition and the properties of the Kullback-Leibler divergence. Where is it used?

KL divergence is the average additional amount of information between q and p.
It is non-symmetric and non-negative. It only equals zero when p equals q.
Give an unknown distribution p we want to approximate that with a distribution q. It can be used to measure the difference between two probability distribution. And approximate the posterior in variational methods and EM algorithm.


##### What are the main ideas of mean field and expectation propagation?

Goal: approximate the posterior with proposal distribution q.
Mean field theory: after factorizing the proposal distribution, we optimize each factor one by one. And for each factor, the log of the optimal solution is obtained by taking the expectation w.r.t all other factors of the log-joint probability of complete data.
EP: Assuming q is from exponential family. Also choose one factor from q and remove it from q by division. Then minimize the KL divergence using moment matching (same mean and covariance) for new q. Then evaluate the new factor. Do this until convergence.

##### What is a cavity distribution?

Remove one factor of a distribution by division.

##### What is a conjugate prior?

A conjugate prior is used to solve the overfitting problem in MLE method. It has the same functional form with posterior. And the calculated posterior can be used as new prior for new data.
Such prior is called conjugate to the likelihood.
Binominal distribution  Beta-distribution.
Multinomial distribution  Dirichlet distribution

##### What is a Dirichlet Process and how can we construct one?
A Dirichlet process is a probability distribution of probability distributions G. It is only defined implicitly, we can say whether a given probability measure is sampled from a DP but we can not construct one. It contains two parameters: alpha concentration and H base measure.
A sample from Dirichlet process is an infinite and discrete distribution.
It can be constructed using the stick-breaking analogy:
A stick has the length of 1. Then select a random number beta between 0 to 1 from a Beta-distribution then break the stick at pi=beta*length. Repeat for the rest stick infinitely.

##### What is the idea of the Chinese Restaurant Process?
The restaurant has infinite tables, each with infinite seats.
Everytime a new customer comes in and sits at an occupied table with probability proportional to the number of people. So the probability to choose a new table is decreasing. The rich get richer.

##### What is the principle of Affinity Propagation?
Given a similarity matrix for all data points
To determine cluster centers (exemplars) that explain other data points in an optimal way.
Responsibility: I thinks j
Availability: j thinks itself.
Recomputed responsibilities and availabilities until convergence.
Choose the one maximize the responsibility and availability.

##### How does Spectral Clustering work?
Undirected graph, edge weights are similarities, weighted degree of a node is sum of all outgoing edges.

Given similarity matrix W and K.
Compute D: degree matrix, diagonal with sum of all outgoing weights.
Compute the graph Laplacian: L = D – W
Compute the eigenvectors of L with K smallest eigenvalues. Which means weak connection between clusters.
Treat each vector as K-dim data point and cluster them with K-means clustering.
Original data point in each vector belongs the resulting clusters.

##### What is agglomerative clustering?
Start with N clusters, each contains only one point.
At each step, merge the two most similar groups.
Repeat until there is only one single group.

Linkage:
1.	Single: closest
2.	Complete: farthest.
3.	Average: average.

Evaluation:
1.	Purity: compute the purity and normalize to choose the one that is purest.
2.	Mutual information: compare with groundtruth. Entropy like.
