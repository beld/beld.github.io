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

  [题目链接](https://leetcode.com/problems/binary-tree-inorder-traversal/"Binary Tree Inorder Traversal")
  昨晚直接用了ArrayList去实现树的保存，今天刷这道题时，看到给出的返回类型是List<Integer>，
不解两者区别，直接new List<Integer>的话就会有\"List is abstract; cannot be instantiated\"
的错误，google后弄明白List是ArrayList和LinkedList的interface。
那么下面两种声明方式有什么区别呢？

    List<Integer> list = new ArrayList<Integer>();
    ArrayList<Integer> aList = new ArrayList<Integer>();


list是List对象，只能使用ArrayList中实现了继承了List的部分，ArrayList也继承了继承了Serializable Interface，但是list不可以使用。

树的遍历一般有递归与迭代两种方法，其实都是将一个复杂的问题分解为一次解决一小部分，最后再把结果拼起来。


> 递归 Recursion：

递归将复杂问题一点点分解，直到你能解决为止。

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
> 迭代 Iteration

一直重复直到任务完成，比如循环计数截止或者链表指针为空。
这里用栈来模拟递归过程。

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

以上两张方法都是O(n)的时间复杂度（递归本身占用stack空间），空间复杂度则是递归栈的大小，即O(logn)。

> Morris Traversal

最后还有一种迭代方法，不用栈作为辅助空间，所以空间复杂度为O(1)，即常数空间。
其实使用了线索二叉树，明天再码。
