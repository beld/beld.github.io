---
layout:     post
title:      "LeetCode: Unique Binary Search Trees"
subtitle:   " \"二叉搜索树\""
date:       2015-11-24 02:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - LeetCode

---

[题目链接](https://leetcode.com/problems/unique-binary-search-trees/)

Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
For example,
Given n = 3, there are a total of 5 unique BST's.

```
1         3     3      2      1
 \       /     /      / \      \
  3     2     1      1   3      2
 /     /       \                 \
2     1         2                 3
```

分析递归关系，首先，假定i为当前root结点,结构不同的树的数量记为F(i,n)，所有n种情况总数量为G(n)。
举例来说，当n=7，i=3时，左枝就为[1,2]组成的树G，右枝就为[4,5,6,7]组成的树。
[关键]：右枝[4,5,6,7]组成的树的数量，就相当于[1,2,3,4]的情况。
所以，F(3,7) = G(2) * G(4)。 可以推出，F(i,n) = G(i-1) * G(n-i)

```java
public static int numTrees(int n) {
    int[] G = new int[n+1];
    G[0] = 1;
    G[1] = 1;
    for (int i = 2; i < G.length; i++) {
        for (int j = 0; j < i; j++) {
            G[i] += G[j] *  G[i-1-j];
        }
    }
    return G[n];
}
```
