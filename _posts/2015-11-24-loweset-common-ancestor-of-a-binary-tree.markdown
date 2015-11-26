---
layout:     post
title:      "LeetCode: Lowest Common Ancestor of a Binary Tree"
subtitle:   " \"二叉搜索树的最近公共祖先\""
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

[题目链接](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

```
           ______3______
          /             \
      ___5__          ___1__
     /      \        /      \
    6        2      0        8
            /  \
           7    4
```

#### 递归 Recursion

代码如下：

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {
        return root;
    } else {
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null && right == null) {
            return null;
        } else if (left != null && right == null) {
            return left;
        } else if (left == null && right != null) {
            return right;
        } else {
            return root;
        }
    }
}
```


#### 非递归，迭代，Iteration
