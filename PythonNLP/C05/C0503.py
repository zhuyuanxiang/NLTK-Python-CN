# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0503.py
@Version    :   v0.1
@Time       :   2020-11-20 9:59
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# 5.3 使用Python字典映射词及其属性(P206)
# Python字典数据类型（以称为关联数组或者哈希数组），学习如何使用字典表示包括词性在内的各种不同语言信息
# 5.3.1 索引链表 与 字典 的区别

# 5.3.2. Python字典
pos = {}
pos['colorless'] = 'ADJ'
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
print("pos['colorless']= ", pos['colorless'])
print("pos= ", pos)

# 访问不存在的键
pos['green']

# 字典转换成链表
print("list(pos)= ", list(pos))

# 字典排序
print("sorted(pos)= ", sorted(pos))

# 字典顺序访问
word_list = [
        w
        for w in pos
        if w.endswith('s')
]
print("word_list= ", word_list)

# 遍历字典中的数据
# for word in sorted(pos):
for word in pos:
    print(word + ":", pos[word])

# 访问字典的方法
print("键= ", pos.keys())
print("值= ", pos.values())
print("对= ", pos.items())

# 分开获取字典中条目的键和值
# for key, val in sorted(pos.items()):
for key, val in pos.items():
    print(key + ":", val)

# 字典中键必须惟一
pos['sleep'] = 'V'
print("pos['sleep']= ", pos['sleep'])
pos['sleep'] = 'N'
print("pos['sleep']= ", pos['sleep'])

# 5.3.3. 定义字典（创建字典的两种方式）
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
print("pos= ", pos)
pos = dict(colorless='ADJ', ideas='N', sleep='V', furiously='ADV')
print("pos= ", pos)

# 5.3.4. 默认字典（字典创建新键时的默认值）
from collections import defaultdict

# 默认值可以是不变对象
frequency = defaultdict(int)
frequency['colorless'] = 4
print("frequency= ", frequency)
print("frequency['colorless']= ", frequency['colorless'])
# 访问不存在的键时，自动创建，使用定义的默认值
print("frequency['ideas']= ", frequency['ideas'])
print("list(frequency.items())= ", list(frequency.items()))

# 默认值也可以是可变对象
pos = defaultdict(list)
pos['sleep'] = ['NOUN', 'VERB']
print("pos =", pos)
print("pos['sleep']= ", pos['sleep'])
print("pos['ideas']= ", pos['ideas'])
print("list(pos.items())= ", list(pos.items()))


# 默认值为自定义对象
class myObject():
    def __init__(self, data=0):
        self._data = data
        return


oneObject = myObject(5)
print("oneObject._data= ", oneObject._data)
twoObject = myObject()
print("twoObject._data= ", twoObject._data)

pos = defaultdict(myObject)
pos['sleep'] = myObject(5)
print("pos['ideas']= ", pos['ideas'])
print("list(pos.items())= ", list(pos.items()))
print("pos['sleep']._data= ", pos['sleep']._data)
print("pos['ideas']._data= ", pos['ideas']._data)

# 默认 lambda 表达式
pos = defaultdict(lambda: 'NOUN')
pos['colorless'] = 'ADJ'
print("pos['colorless']= ", pos['colorless'])
print("pos['blog']= ", pos['blog'])
print("list(pos.items())= ", list(pos.items()))

# 使用 UNK(out of vocabulary)（超出词汇表）标识符来替换低频词汇
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [
        word
        for (word, _) in vocab.most_common(1000)
]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
print("list(mapping.items())[:20]= ", list(mapping.items())[:20])
alice2 = [
        mapping[v]
        for v in alice
]
print("alice2[:20]= ", alice2[:20])

# 5.3.5. 递增地更新字典

# Ex5-3 递增地更新字典，按值排序
counts = nltk.defaultdict(int)
for (word, tag) in nltk.corpus.brown.tagged_words(categories='news', tagset='universal'):
    counts[tag] += 1
print("counts['NOUN']= ", counts['NOUN'])
print("sorted(counts)= ", sorted(counts))
print("counts= ", counts)

from operator import itemgetter

# IndexError: tuple index out of range
sort_keys = sorted(counts.items(), key=itemgetter(0), reverse=False)
print("sort_keys= ", sort_keys)
sort_keys = sorted(counts.items(), key=itemgetter(1), reverse=False)
print("sort_keys= ", sort_keys)
sort_keys = sorted(counts.items(), key=itemgetter(1), reverse=True)
print("sort_keys= ", sort_keys)
# itemgetter(2) 没有这个选项，没法用于排序
# sort_keys = sorted(counts.items(), key=itemgetter(2), reverse=False)
# print("sort_keys= ", sort_keys)
key_list = [
        t
        for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)
]
print("key_list= ", key_list)

pair = ('NP', 8336)
print("pair= ", pair)
print("pair[1]= ", pair[1])
print("itemgetter(0)(pair)= ", itemgetter(0)(pair))
print("itemgetter(1)(pair)= ", itemgetter(1)(pair))

# 通过最后两个字母索引词汇
last_letters = defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

print("last_letters['ly']= ", last_letters['ly'])
print("last_letters['xy']= ", last_letters['xy'])

# 颠倒字母而成的字（回文构词法，相同字母异序词，易位构词，变位词）索引词汇
anagrams = defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)
print("anagrams['aeilnrt']= ", anagrams['aeilnrt'])
print("anagrams['kloo']= ", anagrams['kloo'])
print("anagrams['Zahity']= ", anagrams['Zahity'])
print("anagrams[''.join(sorted('love'))]= ", anagrams[''.join(sorted('love'))])

# NLTK 提供的创建 defaultdict(list) 更加简便的方法
# nltk.Index() 是对 defaultdict(list) 的支持
# nltk.FreqDist() 是对 defaultdict(int) 的支持（附带了排序和绘图的功能）
anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
print("anagrams['aeilnrt']= ", anagrams['aeilnrt'])

anagrams = nltk.FreqDist(''.join(sorted(w)) for w in words)
print("anagrams.most_common(20)= ", anagrams.most_common(20))

# 5.3.6. 复杂的键和值

# 使用复杂的键和值的默认字典
pos = defaultdict(lambda: defaultdict(int))
brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news', tagset='universal')
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1

print("pos[('DET', 'right')]= ", pos[('DET', 'right')])
print("pos[('NOUN', 'further')]= ", pos[('NOUN', 'further')])
print("pos[('PRT', 'book')]= ", pos[('PRT', 'book')])

# 5.3.7. 颠倒字典

# 表5-5：Python 字典的常用方法
# 通过键查值速度很快，但是通过值查键的速度较慢，为也加速查找可以重新创建一个映射值到键的字典
counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word] += 1

# 通过值查键的一种方法
key_list = [
        key
        for (key, value) in counts.items()
        if value == 32
]
print("key_list= ", key_list)

# 使用键-值对字典创建值-键对字典
# pos 是键-值对字典；pos2 是值-键对字典
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
print("pos= ", pos)
pos2 = dict(
        (value, key)
        for (key, value) in pos.items()
)
print("pos2['N']= ", pos2['N'])

# 一个键有多个值的键-值字典不能使用上面的方法创建值-键字典
# 提供了一个新的方法创建值-键对字典
pos.update({'cats': 'N', 'scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
print("pos= ", pos)
pos2 = defaultdict(list)
for key, value in pos.items():
    pos2[value].append(key)
print("pos2['ADV']= ", pos2['ADV'])

# 使用 nltk.Index() 函数创建新的值-键对字典
pos2 = nltk.Index(
        (value, key)
        for (key, value) in pos.items()
)
print("pos2['ADV']= ", pos2['ADV'])
