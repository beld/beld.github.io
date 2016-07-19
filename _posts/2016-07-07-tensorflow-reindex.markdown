---
layout:     post
title:      "Tensorflow: How to re-index the tensors"
subtitle:   ""
date:       2016-07-07 12:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Tensorflow
---

```Python
import tensorflow as tf
import numpy as np
batch_size = 2
bucket_size = 3
n_dims = 13
t = tf.placeholder(tf.float32, shape=[batch_size, bucket_size, n_dims])

with tf.Session() as sess:
    a = [i+1 for i in range(13)]
    original_input = []
    for batch_idx in range(1, batch_size + 1):
        bucket = []
        for bucket_idx in range(3, bucket_size + 3):
            bucket.append([bucket_idx * batch_idx for i in range(13)])
        original_input.append(np.array(bucket))
    print original_input

    # Numpy re-index to [bucket_size, batch_size, n_dims]
    reindexed_input = []
    for bucket_idx in range(bucket_size):
        reindexed_input.append(
            np.array([original_input[batch_idx][bucket_idx]
                for batch_idx in range(batch_size)])
        )
    print reindexed_input
    t = tf.reshape(t, [bucket_size, batch_size, n_dims])

    tensors = []
    for bucket_i in tf.unpack(t):
        buckets = []
        for batch_i in tf.unpack(bucket_i):
            batchs = []
            for dim_i in tf.unpack(batch_i):
                batchs.append(dim_i)
            buckets.append(batchs)
        tensors.append(buckets)

    reindexed_tensor = []
    for batch_idx in range(batch_size):
        batch_t = []
        for dim_idx in range(n_dims):
            dim_t = []
            for bucket_idx in range(bucket_size):
                dim_t.append(tensors[bucket_idx][batch_idx][dim_idx])
            dim_t = tf.pack(dim_t)
            batch_t.append(dim_t)
        batch_t = tf.pack(batch_t)
        reindexed_tensor.append(batch_t)
    # reindexed_tensor.pack(reindexed_tensor)
    output = sess.run(reindexed_tensor, feed_dict={t:reindexed_input})
    print output
```


```
  [array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
       [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]), array([[ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
       [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]])]
[array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]), array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
       [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]), array([[ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]])]
[array([[ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.],
       [ 3.,  4.,  5.]], dtype=float32), array([[  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.],
       [  6.,   8.,  10.]], dtype=float32)]
```
