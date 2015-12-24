---
layout:     post
title:      "Linear Regression Exercise"
subtitle:   "in MATLAB"
date:       2015-12-24 02:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Machine Learning
---

Stanford Deep Learning [Openclassroom](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=DeepLearning&doc=exercises/ex2/ex2.html)

#### Data

Download [ex2Data.zip](http://openclassroom.stanford.edu/MainFolder/courses/DeepLearning/exercises/ex2materials/ex2Data.zip)
The files contain some example measurements of heights for various boys between the ages of two and eights. The y-values are the heights measured in meters, and the x-values are the ages of the boys corresponding to the heights.
There are 50 training samples to develop a linear regression model.

#### Supervised learning problem
Plot the training set and label the axes

```Matlab
x = load('ex2x.dat');
y = load('ex2y.dat');
figure % open a new figure window
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
```

Before starting gradient descent, we need to add the $$x_0 = 1$$ intercept term to every example. To do this in

```Matlab
m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x
```

#### Linear regression

the linear regression model is
<center>$$h_{\theta}(x) = \theta^Tx = \sum_{i=0}^n \theta_i x_i, \nonumber$$</center>
and the batch gradient descent update rule is
<center>$$\theta _{ j }:=\theta _{ j }-\alpha \frac { 1 }{ m } \sum _{ i=1 }^{ m } (h_{ \theta }(x^{(i)})-y^{(i)}) x_{ j }^{ (i) }\; \; \; \; \;$$ (for all $$j$$)</center>

Implement gradient descent using a learning rate of $$\alpha = 0.07$$. Since Matlab index vectors starting from 1 rather than 0, you'll probably use theta(1) and theta(2) in Matlab/Octave to represent $$\theta_0$$ and $$\theta_1$$.
Continue running gradient descent for 1500 iterations until $\theta$ converges. After convergence, record the final values of $$\theta_0$$ and $$\theta_1$$.

```Matlab
theta = zeros(size(x(1,:)))'; % initialize fitting parameters
MAX_ITR = 1500;
alpha = 0.07;

for num_iteration = 1:1500
    grad = 1/m * x' * ((x * theta) - y);
    theta = theta - alpha * theta;
end
```

When you have found $$\theta$$, plot the straight line fit from your algorithm on the same graph as your training data.

```Matlab
hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta, '-') % remember that x is now a matrix with 2 columns
                           % and the second column contains the time info
legend('Training data', 'Linear regression')
```

Note that for most machine learning problems, $x$ is very high dimensional, so we don't be able to plot $$h_\theta(x)$$. But since in this example we have only one feature, being able to plot this gives a nice sanity-check on our result.
