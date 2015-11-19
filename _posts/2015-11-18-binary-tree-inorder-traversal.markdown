---
layout:     post
title:      "LeetCode: Binary Tree Inorder Traversal"
subtitle:   " \"二叉树的中序遍历——递归与迭代\""
date:       2015-11-18 17:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - 算法
    - Algorithm
    - LeetCode
    - Tree
    - 树
---

  https://leetcode.com/problems/binary-tree-inorder-traversal/
  昨晚直接用了ArrayList去实现树的保存，今天刷这道题时，看到给出的返回类型是List<Integer>，
不解两者区别，直接new List<Integer>的话就会有\"List is abstract; cannot be instantiated\"
的错误，google后弄明白List是ArrayList和LinkedList的interface。
那么下面两种声明方式有什么区别呢？
```java
List<Integer> list = new ArrayList<Integer>();
ArrayList<Integer> aList = new ArrayList<Integer>();
```
list是List对象，只能使用ArrayList中实现了继承了List的部分，ArrayList也继承了继承了Serializable Interface，但是list不可以使用。

树树的

> 递归，Recursion：

```java
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> list = new List<Integer>();
    helper(root, list);
    return list;
}

public void helper(TreeNode p, List<Integer> l) {
    if (p != null) {
        helper(p.left, l);
        l.add(p.val);
        helper(p.right, l);
    } else {
        return;
    }       
}
```
> 迭代, Iteration

迭代就是用
```java
public ArrayList<Integer> inorderTraversal(TreeNode root) {
  ArrayList<Integer> aList = new ArrayList<Integer>();
  LinkedList<TreeNode> stack = new LinkedList<TreeNode>();
  while (root != null || !stack.isEmpty()) {
    if (root != null) {
      stack.push(root);
      root = root.left;
    } else {
      root = stack.pop();
      aList.add(root.val);
      root = root.right;
    }
  }
  return aList;
}
```
