---
layout:     post
title:      "LeetCode: Invert Binary Tree"
subtitle:   " \"反转二叉树\""
date:       2015-11-22 23:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - 算法
    - Algorithm
    - LeetCode
    - Tree
    - 树
---

[题目链接](https://leetcode.com/problems/invert-binary-tree/)

This problem was inspired by this original tweet by Max Howell:

> Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so fuck off.

之前看过这条消息，又去知乎看了看他为什么被拒的原因分析，这么牛的人肯定不会因为这么简单一道题被拒。
书归正传，先来看递归解法。

```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```
-->

```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

#### 递归 Recursion

分析：将左右子树交换后就可以把问题简化成分别对两棵子树再进行翻转。
刚开始考虑了最终条件，将所有条件都列了出来。如下：

```java
public TreeNode invertTree(TreeNode root) {
    if (root == null) {
    } else if (root.left == null && root.right == null) {
    } else if (root.left == null) {
        root.left = root.right;
        root.right = null;
        invertTree(root.left);
        invertTree(root.right);
    } else if (root.right == null) {
        root.right = root.left;
        root.left = null;
        invertTree(root.left);
        invertTree(root.right);
    } else {
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.left);
        invertTree(root.right);
    }
    return root;
}
```
后来看知乎的时候发现有人贴了代码，其实后三种条件可以合并着写，代码更简洁：

```java
public TreeNode invertTree(TreeNode root) {
    if (root != null) {
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.left);
        invertTree(root.right);
    }
    return root;
}
```

#### 非递归，迭代，Iteration

写的应该没有错，但是一直Time Limit Exceeded，弄了一小时无解。
看了别人声明用Queue或者Deque然后用LinkedList实例化，换了都没区别。
最后才发现中间有个地方改代码的时候没把root改为node,就这样睁眼瞎的浪费一小时毫无头绪……

```java
public TreeNode invertTree(TreeNode root) {
    LinkedList<TreeNode> stack = new LinkedList<>();
    if (root == null) {
    } else {
        TreeNode node = root;
        stack.push(node);
        while(!stack.isEmpty()) {
            node = stack.pop();
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
    }
    return root;
}
```
