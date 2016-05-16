---
layout:     post
title:      "Word Embeddings"
subtitle:   ""
date:       2016-04-25 10:30:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Tensorflow
    - Machine Learning
---

Motivation: Why Learn Word Embeddings?

Image and audio processing systems work with rich, high-dimensional datasets encoded as vectors of the individual raw pixel-intensities for image data, or e.g. power spectral density coefficients for audio data. Natural language processing systems traditionally treat words as discrete atomic symbols, and therefore 'cat' may be represented as Id537 and 'dog' as Id143. These encodings are arbitrary, and provide no useful information to the system regarding the relationships that may exist between the individual symbols. Representing words as unique, discrete ids furthermore leads to data sparsity.

Vector space models (VSMs) represent (embed) words in a continuous vector space where semantically similar words are mapped to nearby points ('are embedded nearby each other'). VSM all methods depend in some way or another on the Distributional Hypothesis, which states that words that appear in the same contexts share semantic meaning. Count-based methods compute the statistics of how often some word co-occurs with its neighbor words in a large text corpus, and then map these count-statistics down to a small, dense vector for each word. Predictive models directly try to predict a word from its neighbors in terms of learned small, dense embedding vectors (considered parameters of the model).

Word2vec is a particularly computationally-efficient predictive model for learning word embeddings from raw text. CBOW (Continuous Bag-of-Words) predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). This turns out to be a useful thing for smaller datasets.

Scaling up with Noise-Contrastive Training

ML is expensive to train a normalized neural probabilistic language models because we need to compute and normalize each probability using the score for all other words in the current context, at every training step. On the other hand, for feature learning in word2vec we do not need a full probabilistic model.The CBOW and skip-gram models are instead trained using a binary classification objective (logistic regression) to discriminate the real target words from imaginary (noise) words, in the same context.

<center>$$ J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) + k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right] $$</center>

This objective is maximized when the model assigns high probabilities to the real words, and low probabilities to noise words. Technically, this is called Negative Sampling.

The Skip-gram Model

As an example, let's consider the dataset "the quick brown fox jumped over the lazy dog". We could define 'context' in any way that makes sense. For now, let's stick to the vanilla definition and define 'context' as the window of words to the left and to the right of a target word. Using a window size of 1, we then have the dataset:

([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...

of (context, target) pairs. Recall that skip-gram inverts contexts and targets, and tries to predict each context word from its target word. Therefore our dataset becomes

(quick, the), (quick, brown), (brown, quick), (brown, fox), ...

of (input, output) pairs. The objective function is defined over the entire dataset, but we typically optimize this with stochastic gradient descent (SGD) using one example at a time (or a 'minibatch' of batch_size examples, where typically 16 <= batch_size <= 512).

We select num_noise number of noisy (contrastive) examples by drawing from some noise distribution, typically the unigram distribution, $$P(w)$$. The goal is to make an update to the embedding parameters $$θ$$ to improve (in this case, maximize) this objective function. We do this by deriving the gradient of the loss with respect to the embedding parameters. We then perform an update to the embeddings by taking a small step in the direction of the gradient.

We can visualize the learned vectors by projecting them down to 2 dimensions using for instance something like the t-SNE dimensionality reduction technique.

1. define our embedding matrix: a big random matrix to start.
2. The noise-contrastive estimation loss is defined in terms of a logistic regression model. For this, we need to define the weights and biases for each word in the vocabulary.
3. integerized our text corpus with a vocabulary so that each word is represented as an integer.

'''embed = tf.nn.embedding_lookup(embeddings, train_inputs) '''

look up the vector for each of the source words in the batch.
