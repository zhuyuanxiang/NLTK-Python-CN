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

from tools import *

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1.3 通过简单的统计来计算语言
saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']
print("sorted(saying)= ", sorted(saying))
tokens = set(saying)
print("set(saying)= ", tokens)
tokens = sorted(tokens)
print("tokens= sorted(set(saying))= ", tokens)
print("tokens[-3:]= ", tokens[-3:])

# 1.3.1 频率分布
fdist1 = FreqDist(text1)
print("FreqDist(text1)= ", fdist1)
fdist1

# FreqDist 的切片
print("fdist1.most_common(50)= ", fdist1.most_common(50))

# FreqDist 的索引
print("fdist1['whale']= ", fdist1['whale'])

# 图1-4：50个最常用词的累积频率图
fdist1.plot(50, cumulative=True)

# hapaxes() 找出低频词，只出现一次的单词
print("fdist1.hapaxes()[:50]= ", fdist1.hapaxes()[:50])
print("len(fdist1.hapaxes())= ", len(fdist1.hapaxes()))

# 这个用法已经废弃
vocabulary1 = fdist1.keys()
print("list(vocabulary1)[:50]= ", list(vocabulary1)[:50])

# 1.3.2 细粒度的选择单词
# -   数学公式：$\{w|w \in V \& P(W) \}$
# -   程序代码：`[w for w in V if p(w)]`
V = set(text1)
long_words1 = [w for w in V if len(w) > 15]
print("len(long_words)= ", len(long_words1))
sorted(long_words1)

# 聊天语料库中所有长度超过 7 个字符，并且出现次数超过 7 次的单词
fdist5 = FreqDist(text5)
long_words5 = [w for w in set(text5) if len(w) > 7 and fdist5[w] > 7]
print("len(long_words5)= ", len(long_words5))
sorted(long_words5)

# 1.3.3 词语搭配 和 双连词
# **搭配**：是不经常出现在一起的词序列。例如：『red wine』 是一个搭配，而『the wine』不是
# 要获取搭配，需要从提取文本词汇中的词对(即双连词)，提取双连词使用 bigrams() 函数
doubleWords = list(bigrams(['more', 'is', 'said', 'than', 'done']))
print("doubleWords= ", doubleWords)

# collocations() 寻找出现频率大于基准频率的连词，默认频率=20，默认窗口=2
show_title("text4.collocations()")
print(text4.collocations())
show_title("text8.collocations()")
print(text8.collocations())

# 1.3.4 计算其他东西
print("文本中单词长度序列：", [len(w) for w in text1][:50])

fdist_word_len_1 = FreqDist([len(w) for w in text1])
show_title("查看文本中单词长度的分布")
print(fdist_word_len_1.tabulate())
fdist_word_len_1.plot()

print("文本中单词长度序列中的关键字：", sorted(fdist_word_len_1.keys()))
print("文本中单词长度序列中的低频关键字(即这类单词长度的单词在文本中出现的较少)：", fdist_word_len_1.hapaxes())
print("文本中单词的个数：", fdist_word_len_1.N(), "==", len(text1))

print("文本中 5 个出现频率最高的单词长度序列及其出现次数 =", fdist_word_len_1.most_common(5))
print("文本中的样本总数= ", fdist_word_len_1.N())
print("文本中单词长度序列最多的单词出现次数= ", fdist_word_len_1.max())
print("文本中单词长度为 3 的单词出现次数= ", fdist_word_len_1[3])
print("文本中单词长度为 3 的单词出现占比= ", fdist_word_len_1.freq(3))

# 以频率递减的顺序遍历样本
for i, element in enumerate(fdist_word_len_1.elements()):
    if i < 100:  # 避免消耗过长时间输出数据
        print(element, end=', ')
