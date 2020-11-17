# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0403.py
@Version    :   v0.1
@Time       :   2020-11-17 17:36
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import re

import nltk
# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？
from nltk.corpus import brown

# 4.3 风格的问题(P152)
# 4.3.1 Python代码的风格

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')

cv_words_pairs = [
        (cv, w)
        for w in rotokas_words
        for cv in re.findall('[ptksvr][aeiou]', w)
]
cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre)
)

ha_words = [
        'aaahhhh', 'ah', 'ahah', 'ahahah', 'ahh', 'ahhahahaha',
        'ahhh', 'ahhhh', 'ahhhhhh', 'ahhhhhhhh', 'ha', 'haaa',
        'hah', 'haha', 'hahaaa', 'hahah', 'hahaha'
]

syllables = []


def process(aList):
    # process sth.
    return


if (
        len(syllables) > 4
        and len(syllables[2]) == 3
        and syllables[2][2] in ['a', 'e', 'i', 'o', 'u']
        and syllables[2][3] == syllables[1][3]
):
    process(syllables)

# 4.3.2 过程风格 与 声明风格(P153)
# 统计布朗语料库中词的平均长度
tokens = nltk.corpus.brown.words(categories='news')
count = 0
total = 0
for token in tokens:
    count += 1
    total += len(token)
print('total / count={:.3f}'.format(total / count))

token_list = [
        len(t)
        for t in tokens
]
total = sum(
        len(t)
        for t in tokens
)
print('total / count={:.3f}'.format(total / len(tokens)))

# 对单词排序
print("慢速代码")
word_list = []
i = 0
while i < len(tokens[:75]):
    j = 0
    while j < len(word_list) and word_list[j] <= tokens[i]:
        j += 1
    if j == 0 or tokens[i] != word_list[j - 1]:
        word_list.insert(j, tokens[i])
    i += 1
print(word_list)

# 下面是等效的代码，代码更简洁，速度更快
print("快速代码")
word_list = sorted(set(tokens[:75]))
print(word_list)

# 统计布朗语料库中单词占比数，超过25%后停止输出
fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)  # word在总文本中的占比数
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
        break

# P155 寻找最长的单词(只能找到第一个长度最长的词)
text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
    if len(word) > len(longest):
        longest = word
print('longest word:{}'.format(longest))

# 下面是等效的代码，使用两个链表推导式
# 可以找到所有最长的词
maxlen = max(len(word) for word in text)
print([word for word in text if len(word) == maxlen])

# 4.3.3 计数器（counter）的常规用法
# 使用循环变量来提取链表中连续重叠的3-grams
n = 3
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
print("3-grams= ", [sent[i:i + n] for i in range(len(sent) - n + 1)])
# 下面是等效的代码
print("3-grams= ", list(nltk.trigrams(sent)))
# 下面是 2-grams
print("2-grams= ", list(nltk.bigrams(sent)))
# 下面是 4-grams
print("4-grams= ", list(nltk.ngrams(sent, 4)))

import pprint

# 使用循环变量构建多维结构
# 嵌套的链表推导式
m, n = 3, 7
array = [
        [
                set()
                for i in range(n)
        ]
        for j in range(m)
]
array[2][5].add('Alice')
pprint.pprint(array)

# 链表乘法则会对象复制的影响
array = [[set()] * n] * m
array[2][5].add(7)
pprint.pprint(array)
