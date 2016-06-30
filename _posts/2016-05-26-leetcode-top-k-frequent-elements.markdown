---
layout:     post
title:      "LeetCode: Top K Frequent Elements"
subtitle:   ""
date:       2016-05-26 19:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - LeetCode
    - Python
---

[题目链接](https://leetcode.com/problems/top-k-frequent-elements/)

Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].

```
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = []
        pairs = {}
        for i in nums:
            if i in pairs:
                pairs[i] += 1
            else:
                pairs[i] = 1

        a = sorted(pairs.iteritems(), key=lambda asd:asd[1], reverse=True)
        for i in range(k):
            res.append(a[i][0])

        return res
```

Python2.7+之後的版本，在 collections 庫裡有一種類 Counter 可以用。
詳細的操作方法請參考Counter object

利用 Counter(lst) 可以輕鬆得到一個 Counter實例，裡面已經對 lst 中的元素作過統計了。
之後利用 most_common(k) 方法可以輕鬆得到一個排序過的 list of tuple，而且只會剩下前出現頻率前k高的項目，最後用 list comprehension 取出元素本身:
```
from collections import Counter

def top_k_frequent(lst, k):
    return [key for key, count in Counter(lst).most_common(k)]
```
