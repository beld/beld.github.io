---
layout:     post
title:      "Multimodel Deep Learning & Autoencoder"
subtitle:   ""
date:       2016-07-19 12:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Deep Learning
---

[Ngiam et al. ICML’11 (audio-visual speech recognition)](http://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf)


#### Multimodal Fusion Setting

>> Data from all modalities is available at all phases; this represents the typical setting considered in most prior work in audio-visual speech recognition.

#### Cross Modality Learning

>> Data from multiple modalities is available only during feature learning; during the supervised training and testing phase, only data from a single modality is provided. For this setting, the aim is to learn better single modality representations given unlabeled data from multiple modalities.

#### Shared Representation Learning

>>Different modalities are presented for supervised training and testing. This setting allows us to evaluate if the feature representations can capture correlations across different modalities. Specifically, studying this setting allows us to assess whether the learned representations are modality-invariant.
