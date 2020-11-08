# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0104.py
@Version    :   v0.1
@Time       :   2020-11-08 9:51
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
from nltk.book import *

from tools import *

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1.4 Python 的决策 与 控制
# 1.4.1 条件
# P24 表1-3，数值比较运算符
# [w for w in text if condition]
show_title("sent7")
print(sent7)
show_subtitle("[w for w in sent7 if len(w) < 4]")
print([w for w in sent7 if len(w) < 4])
show_subtitle("[w for w in sent7 if len(w) <= 4]")
print([w for w in sent7 if len(w) <= 4])
show_subtitle("[w for w in sent7 if len(w) == 4]")
print([w for w in sent7 if len(w) == 4])
show_subtitle("[w for w in sent7 if len(w) != 4]")
print([w for w in sent7 if len(w) != 4])

# P25 表1-4，词汇比较运算符
# 表1-4：词汇比较运算符
show_subtitle("单词以'ableness'结尾")
print(sorted([w for w in set(text1) if w.endswith('ableness')]))
show_subtitle("单词中包含'gnt'")
print(sorted([term for term in set(text4) if 'gnt' in term]))
show_subtitle("首字母大写")
print(sorted([item for item in set(text6) if item.istitle()]))
show_subtitle("单词是数字")
print(sorted([item for item in set(sent7) if item.isdigit()]))
show_subtitle("单词中包含 '-' 和 'index'")
print(sorted([w for w in set(text7) if '-' in w and 'index' in w]))
show_subtitle("单词首字母大写，并且单词长度超过10")
print(sorted([wd for wd in set(text3) if wd.istitle() and len(wd) > 10]))
show_subtitle("单词首字母不是小写")
print(sorted([w for w in set(sent7) if not w.islower()]))
show_subtitle("单词中含有 'cie' 或者 'cei'")
print(sorted([t for t in set(text2) if 'cie' in t or 'cei' in t]))

# 1.4.2 操作每个元素
print("text1中每个单词的长度= ", [len(w) for w in text1])
print("text1中每个单词都大写= ", [w.upper() for w in text1])

# 进一步理解 Python 的 「链表推导」
print("len(text1)= ", len(text1))
print("len(set(text1))= ", len(set(text1)))
print("len([word.lower() for word in set(text1)])= ", len([word.lower() for word in set(text1)]))
print("len(set([word.lower() for word in text1]))= ", len(set([word.lower() for word in text1])))
# word.isalpha()：如果 word 没有字母就返回 False
print("len(set([word.lower() for word in text1 if word.isalpha()]))= ",
      len(set([word.lower() for word in text1 if word.isalpha()])))

# 1.4.3 嵌套代码块
# 控制结构
word = 'cat'

if len(word) < 5:
    print(f"单词({word})长度小于 5")

if len(word) >= 5:
    print(f"单词({word})长度大于等于 5")

# 循环结构
for word in ['Call', 'me', 'Ishmael', '.']:
    print(word)

# 条件循环
sent1 = ['Call', 'me', 'Ishmael', '.']
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)

# 更加复杂的条件控制
for token in sent1:
    if token.islower():
        print(token, 'is a lower case word')
    elif token.istitle():
        print(token, 'is a title case word')
    else:
        print(token, "is punctuation")

# 习惯用法组合
tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')
