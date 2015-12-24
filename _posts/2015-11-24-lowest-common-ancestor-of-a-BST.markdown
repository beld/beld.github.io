---
layout:     post
title:      "LeetCode: Lowest Common Ancestor of a Binary Search Tree"
subtitle:   " \"二叉搜索树的最近公共祖先\""
date:       2015-11-24 02:00:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - LeetCode

---

[题目链接](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

```
      ______6______
     /             \
  __2__           __8__
 /      \        /      \
0        4      7        9
        /  \
       3    5
```

#### 递归 Recursion

没有看清题目要求是对二叉搜索树进行寻找，其实是忘记二叉搜索树的定义，就写了对所有二叉树的寻找。但是
效率极慢，不可行，不过accepted了，代码如下：

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    TreeNode temp = root;
    while (containedNode(root, p) && containedNode(root, q)) {
        if (containedNode(root.left, p) && containedNode(root.left, q)) {
            temp = root;
            root = root.left;
        } else {
            temp = root;
            root = root.right;
        }
    }
    return temp;
}

public boolean containedNode (TreeNode root, TreeNode p) {
    if (root == null) {
        return false;
    } else {
        if (root.left == p || root.right == p || root == p) {
            return true;    
        } else {
            return (containedNode(root.left, p) || containedNode(root.right, p));
        }
    }
}
```
二叉搜索树的话就简单多了，通过值判断,让根节点在两个节点中间即可。

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root.val > p.val && root.val > q.val) {
        return lowestCommonAncestor(root.left, p, q);
    } else if (root.val < p.val && root.val < q.val) {
        return lowestCommonAncestor(root.right, p, q);
    } else {
        return root;
    }
}
```

#### 非递归，迭代，Iteration

没什么好说的，很简单。

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    LinkedList<TreeNode> stack = new LinkedList<>();
    stack.push(root);
    while(!stack.isEmpty()) {
        root = stack.pop();
        if (root.val > p.val && root.val > q.val) {
            stack.push(root.left);
        } else if (root.val < p.val && root.val < q.val) {
            stack.push(root.right);
        } else {
        }
    }
    return root;
}
```
