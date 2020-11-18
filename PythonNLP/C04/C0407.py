# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0407.py
@Version    :   v0.1
@Time       :   2020-11-18 16:13
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import pprint
import re

import nltk


# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？

# 4.7 算法设计(P175)
# 4.7.1 递归与迭代(P176)
# 两种解决方案各有利弊。递归更容易理解，迭代速度更快


def factorial1(n):
    result = 1
    for i in range(n):
        result *= (i + 1)
    return result


def factorial2(n):
    if n == 1:
        return 1
    else:
        return n * factorial2(n - 1)


print("factorial1(10)= ", factorial1(10))
print("factorial2(10)= ", factorial2(10))


def size1(s):
    return 1 + sum(size1(child) for child in s.hyponyms())


def size2(s):
    layer = [s]
    total = 0
    while layer:
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total


from nltk.corpus import wordnet as wn

dog = wn.synset('dog.n.01')
print("size1(dog)= ", size1(dog))
print("size2(dog)= ", size2(dog))


# Ex4-6 构建一个字母查找树
# 一个递归函数建立一个嵌套的字典结构，每一级嵌套包含给定前缀的所有单词
# 而子查找树含有所有可能的后续词
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value


trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylist')
trie = dict(trie)
pprint.pprint(trie, width=40)
print("trie['c']['h']['a']['t']['value']=", trie['c']['h']['a']['t']['value'])


# 4.7.2 空间与时间的平衡(P179)
# Ex4-7: 一个简单的全文检索系统
# 通过对文档索引集合，提高搜索速度，然后再对文档展开搜索，减少搜索准备
def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub(r'\s+', ' ', contents)
    return contents


def snippet(doc, term):
    text = ' ' * 30 + raw(doc) + ' ' * 30
    pos = text.index(term)
    return text[pos - 30:pos + 30]


print('Building Index...')
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index(
        (w, f)
        for f in files
        for w in raw(f).split()
)

query = ''
while query != 'quit':
    query = input('query> ')
    if query in idx:
        for i, doc in enumerate(idx[query]):
            if i < 10:
                print(snippet(doc, query))
    else:
        print('Not found')


# Ex4-8 预处理已经标的语料库数据，将所有的词和标都转换成整数
def preprocess(tagged_corpus):
    words = set()
    tags = set()
    for sent in tagged_corpus:
        for word, tag in sent:
            words.add(word)
            tags.add(tag)
    wm = dict((w, i) for (i, w) in enumerate(words))
    tm = dict((t, i) for (i, t) in enumerate(tags))
    return [[(wm[w], tm[t]) for (w, t) in sent] for sent in tagged_corpus]


# 使用timeit模块检测执行速度
# Timer类有两个参数：一个是多次执行的代码；一个是只在开始执行一次的设置代码。
# 例子：整数的链表 和 整数的集合 模拟10万个项目的词汇表
# 测试声明将产生随机项，它有50%的机会出现在词汇表中。
from timeit import Timer

vocab_size = 100000
setup_list = 'import random; vocab=range(%d)' % vocab_size
setup_set = 'import random; vocab=set(range(%d))' % vocab_size
statement = 'random.randint(0, %d) in vocab' % (vocab_size * 2)
# 以前的Python集合比列表快，现在几乎没有差别
print(Timer(statement, setup_list).timeit(1000))
print(Timer(statement, setup_set).timeit(1000))
vocab = range(vocab_size)


# 4.7.3 动态规划(P181)
# 动态规划是在自然语言处理中广泛使用的算法。
# -   解决的问题内部包含了多个重叠的子问题。
# -   算法可以避免重复计算这些子问题，而是简单地将它们的计算结果存储在一个查找表中。

# Ex4-9 4种方法计算梵文旋律：迭代、自底向上的动态规划、自上而下的动态规划、内置默记法
# ToDo: 迭代算法，值得反复学习
# 递归计算中会有重复计算的部分
def virahanka1(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka1(n - 1)]
        l = ["L" + prosody for prosody in virahanka1(n - 2)]
        return s + l


virahanka1(4)


# 自底向上的动态规划
# 将较小的实例计算结果填充到表格中，一旦得到感兴趣的值就停止，
# 原则是解决较大问题之前先解决较小的问题，这就是自下而上的动态规划
# 某些计算得到的子问题在解决主问题时可能并不需要，从而造成浪费
def virahanka2(n):
    lookup = [[""], ["S"]]
    for i in range(n - 1):
        s = ["S" + prosody for prosody in lookup[i + 1]]
        l = ["L" + prosody for prosody in lookup[i]]
        lookup.append(s + l)
    return lookup[n]


virahanka2(4)


# 自上而下的动态规划：可以避免计算不需要的子问题带来的浪费
def virahanka3(n, lookup={0: [""], 1: ["S"]}):
    if n not in lookup:
        s = ["S" + prosody for prosody in virahanka3(n - 1)]
        l = ["L" + prosody for prosody in virahanka3(n - 2)]
        lookup[n] = s + l
    return lookup[n]


virahanka3(4)

# 内置默记法：使用Python的装饰器模式引入的缓存机制
from nltk import memoize


@memoize
def virahanka4(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n - 1)]
        l = ["L" + prosody for prosody in virahanka4(n - 2)]
        return s + l


virahanka4(4)

# 由functools.lru_cache实现的Python的memoization比我们的专用memoize函数更全面，就像你在CPython源代码中看到的一样
from functools import lru_cache


@lru_cache(1000)
def virahanka5(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n - 1)]
        l = ["L" + prosody for prosody in virahanka4(n - 2)]
        return s + l


virahanka5(4)
