---
layout:     post
title:      "LeetCode: Same Tree II"
subtitle:   " \"Revision\""
date:       2015-11-20 16:30:00
author:     "Beld"
header-img: "img/post-bg-tree.png"
tags:
    - LeetCode
---

[题目链接](https://leetcode.com/problems/same-tree/)

上次Same Tree用了两个辅助ArrayList空间保存遍历结果再比较，实在是不吝代价完成目标的。
这次更新标准的递归和迭代方法。

#### 递归 Recursion

思路很简单就是分成左右两边各自比较。
但是写的时候if-else逻辑结构不清，最后忘记还有个else分支。
代码如下：

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) {
        return true;
    } else if (p == null || q == null) {
        return false;
    } else {
        if (p.val != q.val) {
            return false;
        } else {
            if (isSameTree(p.left, q.left) && isSameTree(p.right,q.right)) {
                return true;
            } else {
              return false;
            }
        }
    }
}
```
看了别人的代码，发现可以更简洁，逻辑也更清楚了，如下：

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) {
        return true;
    } else if (p == null || q == null) {
        return false;
    }
    return (p.val == q.val) && isSameTree(p.left, q.left)
                            && isSameTree(p.right,q.right);
}
```

#### 非递归，迭代，Iteration

用栈来模拟递归，其实也很简单，但是代价就是空间。
另外，自己写的逻辑虽然没错，但是看上去比较冗余，应该把错误条件都写着前面，然后再进行正确操作。

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    LinkedList<TreeNode> stackP = new LinkedList<TreeNode>();
    LinkedList<TreeNode> stackQ = new LinkedList<TreeNode>();
    stackP.push(p);
    stackQ.push(q);
    while(!stackP.isEmpty() && !stackQ.isEmpty()) {
        p = stackP.pop();
        q = stackQ.pop();
        if(p != null && q != null) {
            if (p.val != q.val) {
                return false;
            } else {
                stackP.push(p.left);
                stackP.push(p.right);
                stackQ.push(q.left);
                stackQ.push(q.right);
            }
        } else if ((p == null && q != null) || (q == null && p != null)) {
            return false;
        }
    }
    return true;
}
```
