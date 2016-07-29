---
layout:     post
title:      "Layer Normalization"
subtitle:   ""
date:       2016-07-19 12:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Deep Learning
---


[Layer Normalization](http://arxiv.org/abs/1607.06450)

#### Batch Normalization Drawbacks

1. It normalizes the summed inputs to each hidden unit over the training cases. It is typically impractical to compute the expectations, since it would require forward passes through the whole training dataset with the current set of weights. Instead, µ and σ are estimated using the empirical samples from the current mini-batch. This puts constraints on the size of a mini-batch and it is hard to apply to recurrent neural networks.
2. However, the summed inputs to the recurrent neurons in a recurrent neural network (RNN) often vary with the length of the sequence so applying batch normalization to RNNs appears to require different statistics for different time-steps. The recent sequence to sequence models utilize compact recurrent neural networks to solve sequential prediction problems. It is common among the NLP tasks to have different sentence lengths for different training cases. This is easy to deal with in an RNN because the same weights are used at every time-step. But when we apply batch normalization to an RNN in the obvious way, we need to to compute and store separate statistics for [[each time step]] in a sequence. This is problematic if a test sequence is longer than any of the training sequences. Layer normalization does not have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step. It also has only one set of gain and bias parameters shared over all time-steps
3. Furthermore, batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the mini batches have to be small.


#### Layer Normalization

Unlike batch normalization, the proposed method directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases.

This suggests the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer.

All the hidden units in a layer share the same normalization terms µ and σ, but different training cases have different normalization terms. Unlike batch normalization, layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1.
