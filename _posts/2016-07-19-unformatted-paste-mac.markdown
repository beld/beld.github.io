---
layout:     post
title:      "Mac Office Word 2016修改默认粘贴格式"
subtitle:   ""
date:       2016-07-19 23:00:00
author:     "Beld"
header-img: "img/post-bg-ml.png"
tags:
    - Mac
---

Mac Office默认粘贴原格式，但是绝大多数情况我们都需要的是无格式的粘贴，所以每次粘贴后都要修改一下太令人抓狂。下面说下怎么修改：

#### 1. 创建新的macro

在Tools > Macro > Macros… 里创建一个新的macro叫做PasteUnformatted，然后会打开Visual Basic editor，将下面代码贴入：


```
Sub PasteUnformatted()
Selection.PasteSpecial DataType:=wdPasteText
End Sub
```

#### 2. 重新映射Cmd+V

在Tools > Customize Keyboard… 中选择Macros, 找到PasteUnformatted后，把快捷键Cmd+V赋值给它，然后确认。

#### 3. 重新打开

前两步之后不会立刻生效，需要关闭重新打开。

Enjoy it!
