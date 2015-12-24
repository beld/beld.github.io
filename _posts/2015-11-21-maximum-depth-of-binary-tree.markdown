---
layout:     post
title:      "LeetCode: Maximum Depth of Binary Tree"
subtitle:   " \"二叉树的层数\""
date:       2015-11-21 02:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - LeetCode

---

[题目链接](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

题目很简单，但是却卡在这里。因为不知道max函数，直接码的时候就写了三目运算符，但是超时了。
然后就用了两个int来存一下，结果声明为了全局变量，结果就死活不对，最后才发现。
可能刚游泳回来太累需要早睡吧，不能头脑不清的时候刷题，而且要学会放下，一会再重来。

#### 递归 Recursion

```java
public static int maxDepth(TreeNode root) {
    if (root == null) {
        return 0;
    } else {
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

#### 非递归，迭代，Iteration

真的要困了就睡……想不出来

```java

```
