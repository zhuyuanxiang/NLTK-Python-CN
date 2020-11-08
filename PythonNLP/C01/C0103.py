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
    if i<100:   # 避免消耗过长时间输出数据
        print(element, end=', ')

# 4. Python 的决策 与 控制
# 4.1. 条件
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
sorted([w for w in set(text1) if w.endswith('ableness')])  # 单词以'ableness'结尾
sorted([term for term in set(text4) if 'gnt' in term])  # 单词中包含'gnt'
sorted([item for item in set(text6) if item.istitle()])  # 首字母大写
sorted([item for item in set(sent7) if item.isdigit()])  # 单词是数字

sorted([w for w in set(text7) if '-' in w and 'index' in w])
sorted([wd for wd in set(text3) if wd.istitle() and len(wd) > 10])
sorted([w for w in set(sent7) if not w.islower()])
sorted([t for t in set(text2) if 'cie' in t or 'cei' in t])

# 4.2. 操作每个元素
[len(w) for w in text1]
[w.upper() for w in text1]

len(text1)
len(set(text1))
len([word.lower() for word in set(text1)])
len(set([word.lower() for word in text1]))
len(set([word.lower() for word in text1 if word.isalpha()]))

# 4.3. 嵌套代码块
word = 'cat'

if len(word) < 5:
    print('word length is less than 5')

if len(word) < 5:
    print('word length is greater than or equal to 5')

for word in ['Call', 'me', 'Ishmael', '.']:
    print(word)

# 条件循环
sent1 = ['Call', 'me', 'Ishmael', '.']
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)

tricky = sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')

# 5. 自然语言理解的自动化处理
# 5.1. 词意消歧：相同的单词在不同的上下文中指定不同的意思
# 5.2. 指代消解：检测动词的主语和宾语
# 指代消解：确定代词或名字短语指的是什么？
# 语义角色标注：确定名词短语如何与动词想关联（如代理、受事、工具等）
# 5.3. 自动生成语言：如果能够自动地解决语言理解问题，就能够继续进行自动生成语言的任务，例如：自动问答和机器翻译。
# 5.4. 机器翻译：
# 文本对齐：自动配对组成句子
# 5.5. 人机对话系统：图1-5，简单的话音对话系统的流程架构
# 5.6. 文本的含义：文本含义识别（Recognizing Textual Entailment, RTE）
# 5.7. NLP的局限性
