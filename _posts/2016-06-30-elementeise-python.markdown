---
layout:     post
title:      "Python: Element-wise Addition of 2 Lists"
subtitle:   ""
date:       2016-06-30 23:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Python
---

#### Timing Comparisons

```Python
In [2]: list2 = [4, 5, 6]*10**5
In [3]: list1 = [1, 2, 3]*10**5
In [4]: %timeit from operator import add; map(add, list1, list2)
10 loops, best of 3: 19.3 ms per loop

In [5]: %timeit from itertools import izip; [a + b for a, b in izip(list1, list2)]
10 loops, best of 3: 23.9 ms per loop

In [6]: %timeit [a + b for a, b in zip(list1, list2)]
10 loops, best of 3: 44.7 ms per loop

In [7]:  %timeit from itertools import izip;[sum(x) for x in izip(list1, list2)]10 loops, best of 3: 56.8 ms per loop

In [8]: %timeit [sum(x) for x in zip(list1, list2)]
10 loops, best of 3: 72.1 ms per loop

In [9]: import numpy as np

In [10]: %timeit np.array(list1) + np.array(list2)
10 loops, best of 3: 26.1 ms per loop
```

#### And what is the difference between zip and itertools.izip?
zip computes all the list at once, izip computes the elements only when requested.

One important difference is that 'zip' returns an actual list, 'izip' returns an 'izip object', which is not a list and does not support list-specific features (such as indexing):
