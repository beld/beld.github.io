---
layout:     post
title:      "LeetCode: Unique Binary Search Trees"
subtitle:   " \"二叉搜索树\""
date:       2015-11-24 02:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - 算法
    - Algorithm
    - LeetCode
    - Tree
    - 树
---

[题目链接](https://leetcode.com/problems/count-complete-tree-nodes/)

Given a complete binary tree, count the number of nodes.

>Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

```
         1
       /    \
      2      3
    /   \   /  \
   4     5 6    7
 /  \   /
8    9 10
```
其实不难，直接写了个迭代的方法，时间复杂度O(n)，可是超时了。
偷懒直接看了别人的思路，用了二分查找的方法，如果是一边是满树，直接可以计算。

```java
public int countNodes(TreeNode root) {
    if (root == null) {
        return 0;
    } else {
        int left = leftHeight(root);
        int right = rightHeight(root);
        if (left == right) {
            return 1 << left;
        } else {
            return 1 + countNodes(root.left) + countNodes(root.right);
        }
    }
}

private int leftHeight(TreeNode root) {
    if (root == null) {
        return 0;
    } else {
        return leftHeight(root.left) + 1;
    }
}

private int rightHeight(TreeNode root) {
    if (root == null) {
        return 0;
    } else {
        return rightHeight(root.right) + 1;
    }
}
```

结果上面的方法依旧超时，计算树高时用的递归，看别人用的while循环就可以通过，就改正了下，ac,代码如下：
时间复杂度是O(log^2 n)。其实自己并不清楚时间复杂度的计算，需要研究下专门写一篇。
```java
public int countNodes(TreeNode root) {
    int leftHeight = 0, rightHeight = 0;
    TreeNode left = root, right = root;
    while (left != null) {
        leftHeight += 1;
        left = left.left;
    }
    while (right != null) {
        rightHeight += 1;
        right = right.right;
    }
    if (leftHeight == rightHeight) {
        return (1 << leftHeight) - 1;
    } else {
        return 1 + countNodes(root.left) + countNodes(root.right);
    }
}
```
