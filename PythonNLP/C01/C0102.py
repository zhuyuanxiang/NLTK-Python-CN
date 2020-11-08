# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0102.py
@Version    :   v0.1
@Time       :   2020-11-08 9:51
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
from nltk.book import *

from func import lexical_diversity
from tools import *

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1.2 Python 将文本当作词链表
# 1.2.1 链表
sent1 = ['Call', 'me', 'Ishmael', '.']
print("sent1= ", sent1)
print("sent1 的单词个数：", len(sent1))
print("sent1 中每个单词被使用的次数的平均数：", lexical_diversity(sent1))

print("sent2= ", sent2)
print("sent1+sent2= ", sent1 + sent2)

# 注意append() 函数返回的是运行效果，不是字符串
sent1 = ['Call', 'me', 'Ishmael', '.']
print("sent1= ", sent1)
show_subtitle("sent1.append('Some')")
sent1.append('Some')
print(sent1)

sent1 = ['Call', 'me', 'Ishmael', '.']
print("sent1= ", sent1)
sent1.append(sent2)
show_subtitle("sent1.append(sent2)")
print(sent1)

# 1.2.2 索引列表
# 索引列表
print("text4[173]= ", text4[173])
print("text4.index('awaken')= ", text4.index('awaken'))

# 索引切片
show_subtitle("text5[16715:16735]")
print(text5[16715:16735])
show_subtitle("text6[1600:1625]")
print(text6[1600:1625])

# 索引从零开始
sent = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10']
print("sent[0]= ", sent[0])
print("sent[9]= ", sent[9])
# print("sent[10]= ", sent[10])

# 索引切片
print("sent[5:8]= ", sent[5:8])
print("sent[5]= ", sent[5])
print("sent[7]= ", sent[7])

# 索引切片
print("sent[:3]= ", sent[:3])
print("sent[-1:]= ", sent[-1:])
print("sent[:-1]= ", sent[:-1])

# 索引切片
print("text2[141525:]= ", text2[141525:])

# 索引大小
sent = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10']
sent[0] = 'First'
sent[9] = 'Last'
print("len(sent): ", len(sent))
print("sent[1:9]= ", sent[1:9])

# 索引大小
sent[1:9] = ['Second', 'Third']
print("sent= ", sent)
print("len(sent)= ", len(sent))
# print("sent[9]= ", sent[9])

# 1.2.3 变量
# 链表变量与赋值
sent1 = ['Call', 'me', 'Ishmael', '.']
my_sent = ['Bravely', 'bold', 'Sir', 'Robin', ',', 'rode']
noun_phrase = my_sent[1:4]
print("noun_phrase= ", noun_phrase)

# 链表排序：大写字母 排在 小写字母 前面
w0rDs = sorted(noun_phrase)
show_subtitle("sorted(noun_phrase)")
print(w0rDs)

# 变量名不能使用 Python 的保留字
# not='Camelot'

vocab = set(text1)
vocab_size = len(vocab)
show_subtitle("len(set(text1))")
print("vocab_size= ", vocab_size)

# 1.2.4 字符串
# 字符串的访问与链表相同。

# 字符串处理
name = 'Monty'
# name[0] = 'm'  # TypeError: 'str' object does not support item assignment
print("name= ", name)
print("name[0]= ", name[0])
print("name[:4]= ", name[:4])
print("name*2= ", name * 2)
print("name+'!'= ", name + '!')

# 字符串函数
print("' '.join(['Monty', 'Python'])= ",' '.join(['Monty', 'Python']))
print("'Monty Python'.split()= ", 'Monty Python'.split())