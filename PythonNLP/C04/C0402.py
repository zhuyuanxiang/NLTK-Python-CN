# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0402.py
@Version    :   v0.1
@Time       :   2020-11-16 13:11
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# 4.2 序列（Sequences）(P147)
# 序列对象共有三种（字符串、链表、元组)
# 元组的定义与使用
from tools import show_subtitle

# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？

t1 = 'walk', 'fem', 3  # 定义元组的标准方法
t2 = ('walk', 'fem', 3)  # 括号并非定义元组的标准方法，括号是Python语法的一般功能，是用于分组的
print("'t1 == t2'= ", t1 == t2)
print("t1[0]= ", t1[0])
print("t2[1:]= ", t2[1:])
print("len(t1)= ", len(t1))

t3 = ['a', 'b', 'c'], 'a', 'b', 'c'
t4 = (['a', 'b', 'c'], 'a', 'b', 'c')
print("'t3 == t4'= ", t3 == t4)

emtpy_tuple = ()  # 空元组的定义
print(emtpy_tuple)

single_tuple = 'love',  # 单个元素的元组的定义
print(single_tuple)
print(single_tuple[0])

# spectroroute这个单词查不到
raw = 'I turned off the spectroroute'  # 字符串
text = ['I', 'turned', 'off', 'the', 'spectroroute']  # 链表
pair = (6, 'turned')  # 元组
print('raw[2]={}, text[3]={}, pair[1]={}'.format(raw[2], text[3], pair[1]))
print('raw[-3:]={}, text[-3:]={}, pair[-3:]={}'.format(raw[-3:], text[-3:], pair[-3:]))
print('len(raw)={}, len(text)={}, len(pair)={}'.format(len(raw), len(text), len(pair)))

# 定义集合
# 集合不可以索引访问
sets = set(raw)
print("sets= ", sets)
print("len(set(raw))= ", len(sets))
print("'f' in sets= ", 'f' in sets)

lists = list(sets)
print("lists= ", lists)
print("len(lists)= ", len(sets))
print("'f' in lists= ", 'f' in lists)

sets = set(text)
print("sorted(set(text))= ", sorted(sets))

# 4.2.1 序列类型上的操作(P148)
# 表4-1：遍历序列的各种方式
raw = 'Red lorry, yellow lorry, red lorry, yellow lorry'
print("set(raw)= ", set(raw))  # 不能出现重复的元素
print("list(raw)= ", list(raw))  # 可以出现重复的元素

from nltk import word_tokenize

text = word_tokenize(raw)  # 分词
fdist = nltk.FreqDist(text)  # 词频统计
fdist.tabulate()
print("sorted(fdist)= ", sorted(fdist))
for key in fdist:
    print(key + ':', fdist[key])

raw = 'I turned off the spectroroute'
text = word_tokenize(raw)
print("word_tokenize(raw)= ", text)

words = ['I', 'turned', 'off', 'the', 'spectroroute']
# 利用元组批量赋值
words[2], words[3], words[4] = words[3], words[4], words[2]
print("words= ", words)

words = ['I', 'turned', 'off', 'the', 'spectroroute', '!']
tags = ['noun', 'verb', 'prep', 'det', 'noun']

# zip() 取出两个或者两个以上的序列中的项目，将它们“压缩”打包成单个的配对链表
# 数目不匹配的部分直接丢弃，例如：words比赛tags多一个，就直接丢弃了。
print("list(zip(words, tags))= ", list(zip(words, tags)))
print("list(zip(words, tags, tags))= ", list(zip(words, tags, tags)))

# enumerate() 返回一个包含索引及索引处所在项目的配对。
print("list(enumerate(words))= ", list(enumerate(words)))

for a, b in enumerate(words):
    print('a:{}\tb:{}'.format(a, b))

# 分割数据集为（训练集+测试集)
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
print("len(training_data) / len(test_data)= ", len(training_data) / len(test_data))

# 使用split()函数分词
raw = 'I turned off the spectroroute'
words = raw.split()
print("words= ", words)
wordlens = [(len(word), word) for word in words]
print("wordlens= ", wordlens)

wordlens.sort()
show_subtitle("wordlens.sort()")
print("wordlens= ", wordlens)

wordlens.reverse()
show_subtitle("wordlens.reverse()")
print("wordlens= ", wordlens)

wordlens.pop()
show_subtitle("wordlens.pop()")
print("wordlens= ", wordlens)

print("' '.join(w for (_, w) in wordlens)= ",
      ' '.join(
              w
              for (_, w) in wordlens
      )
      )

# 元组是不可修改的，而链表是可以修改的
lexicon = [('the', 'det', ['Di:', 'D@']), ('off', 'prep', ['Qf', 'O:f'])]
print("lexicon= ", lexicon)

lexicon.sort()
show_subtitle("lexicon.sort()")
print("lexicon= ", lexicon)

lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
show_subtitle("lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])")
print("lexicon= ", lexicon)

del lexicon[0]
show_subtitle("del lexicon[0]")
print("lexicon= ", lexicon)

# 链表转换成元组后，下面的操作都不可执行
lexicon = tuple(lexicon)
# lexicon.sort()
# lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
# del lexicon[0]

# 使用元组还是链表主要还是取决于项目的内容是否与它的位置相关

# 4.2.2 产生器表达式(P151)
# 使用列表推导式方便处理文本
text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone, 
"it means just what I choose it to mean - neither more nor less."'''
print("text= ", text)

words = [
        w.lower()
        for w in nltk.word_tokenize(text)
]
print("words= ", words)

print("words[0]= ", words[0])
print("words[7]= ", words[7])
# ToDo: 句子开头和结尾的双引号是有区别的
print("'words[0] == words[7]'= ", words[0] == words[7])

# 这两种方式都会先生成链表对象，再调用 max() 函数，
# 会占用大量的存储空间
print("max(words)= ", max(words))
max_words = max([
        w.lower()
        for w in nltk.word_tokenize(text)
])
print("max_words= ", max_words)

# max()函数调用时使用了产生器表达式，
# 不仅仅是标记方便，即不仅仅是省略了方括号[]，
# 还利用数据流输入到调用函数中，避免先生成链表对象，也就不会占用过多的存储空间
max_words = max(
        w.lower()
        for w in nltk.word_tokenize(text)
)
print("max_words= ", max_words)
