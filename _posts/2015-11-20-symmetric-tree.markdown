---
layout:     post
title:      "LeetCode: Symmetric Tree"
subtitle:   " \"对称树的判定\""
date:       2015-11-20 23:30:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - 算法
    - Algorithm
    - LeetCode
    - Tree
    - 树
---

[题目链接](https://leetcode.com/problems/symmetric-tree/)

题目很简单，基本与same tree一样，稍加改动即可。

#### 递归 Recursion

忘记了对空树的判断。

```java
public boolean isSymmetric(TreeNode root) {
    if (root == null) {
        return true;
    } else {
        return isSymmetricTree(root.left, root.right);
    }
}

public boolean isSymmetricTree(TreeNode left, TreeNode right) {
    if (left == null && right == null) {
        return true;
    } else if (left == null || right ==null ) {
        return false;
    } else {
       return (left.val == right.val) && isSymmetricTree(left.left, right.right)
                                      && isSymmetricTree(left.right, right.left);
    }
}
```

#### 非递归，迭代，Iteration


```java
public boolean isSymmetric(TreeNode root) {
    if (root == null) {
        return true;
    } else {
        return isSymmetricTree(root.left, root.right);
    }
}

public boolean isSymmetricTree(TreeNode left, TreeNode right) {
    LinkedList<TreeNode> stackL = new LinkedList<TreeNode>();
    LinkedList<TreeNode> stackR = new LinkedList<TreeNode>();
    stackL.push(left);
    stackR.push(right);
    while(!stackL.isEmpty() && !stackR.isEmpty()) {
        left = stackL.pop();
        right = stackR.pop();
        if (left == null) {
            if (right != null) {
                return false;
            } else continue;
        } else if (right == null) {
            return false;
        } else {
            if (left.val != right.val) {
                return false;
            } else {
                stackL.push(left.left);
                stackR.push(right.right);
                stackL.push(left.right);
                stackR.push(right.left);
            }
        }
    }
    return true;
}
```
