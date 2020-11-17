# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0404.py
@Version    :   v0.1
@Time       :   2020-11-17 18:02
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？

# 4.4 结构化编程的基础(P156)
# Ex4-1: 从文件中读取文本
import re


def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file, encoding='utf-8').read()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('\s+', '', text)
    return text


help(get_text)
contents = get_text('output.txt')


# 4.4.1 函数的输入与输出
# 有参数的函数
def repeat(msg, num):
    return ' '.join([msg] * num)


monty = 'Monty Python'
print(repeat(monty, 3))


# 无参数的函数
def monty():
    return 'Monty Python'


print("monty()= ", monty())
print("repeat(monty(), 3)= ", repeat(monty(), 3))
print("repeat('Monty Python', 3)= ", repeat('Monty Python', 3))


# 没有返回值，调用者传入参数，就是请求函数帮助对输入参数排序
def my_sort1(mylist):
    mylist.sort()


# 返回值是排序后的结果，传入的参数没有被改变
def my_sort2(mylist):
    return sorted(mylist)


# 这个函数是危险的，因为输入值已经被修改，但是没有明确地通知调用者
def my_sort3(mylist):
    mylist.sort()
    return mylist()


# 4.4.2 参数传递(P159)
# 第一个传入参数在函数内部被改变了，但是调用者的参数内容没有被改变，因为是按内容传值的
# 第二个传入参数在函数内部被改变了，调用的参数内容也被改变了，因为是按地址传值的
def set_up(word, properties):
    word = 'lolcat'
    properties.append('noun')
    properties = 5


# w没有被改变，p被改变了
w = ''
p = []
print("w= ", w)
print("p= ", p)
set_up(w, p)
print("w= ", w)
print("p= ", p)

# w没有被改变，因为 word 被两次赋值，没有指向 w 的地址
w = ''
word = w
word = 'lolcat'
print("w= ", w)

# p被改变了，因为 properties 没有被再次赋值，而是在原地址访问的链表内追加数据
# 当 properties 被再次赋值时，p 已经被改变
p = []
properties = p
print("p= ", p)
properties.append('noun')
print("p= ", p)
print("properties= ", properties)
properties = 5
print("p= ", p)
print("properties= ", properties)


# 4.4.3 变量的作用域
# 名称解析的LGB规则（本地（local）、全局（global）、内置（built-in））

# 4.4.4 参数类型检查(P160)
# 没有参数类型检查的函数
def tag(word):
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'


print("tag('the')= ", tag('the'))
print("tag('knight')= ", tag('knight'))
# 传入链表后，函数返回值是错误的
print("tag(['Tis', 'but', 'a', 'scratch'])= ", tag(['Tis', 'but', 'a', 'scratch']))


# 使用断言进行参数类型检查
def tag(word):
    assert isinstance(word, str), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'


print("tag('the')= ", tag('the'))
print("tag('knight')= ", tag('knight'))
# 传入链表后，函数断言失败
print("tag(['Tis', 'but', 'a', 'scratch'])= ", tag(['Tis', 'but', 'a', 'scratch']))


# 4.4.5 功能分解
# 使用函数提高程序的可读性和可维护性
def load_corpus():
    return -1


def analyze(data):
    return -1


def present(results):
    return


data = load_corpus()
results = analyze(data)
present(results)

import nltk
from urllib import request
from bs4 import BeautifulSoup

constitution = "http://www.archives.gov/exhibits/charters/constitution_transcript.html"


# Ex4-2 计算高频词的拙劣函数，存在的几个问题：
# 1）修改了第二个参数的内容
# 2）输出了已经计算过的结果
def freq_words(url, freqdist, n):
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html, 'html.parser').get_text()
    for word in nltk.word_tokenize(text):
        freqdist[word.lower()] += 1
    result = []
    for word, count in freqdist.most_common(n):
        result += [word]
    print(result)


fd = nltk.FreqDist()
freq_words(constitution, fd, 30)


# 重构Ex4-2该函数，得到Ex4-3 用来计算高频词的函数
def freq_words(url, n):
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html, 'html.parser').get_text()
    fd = nltk.FreqDist(
            word.lower()
            for word in nltk.word_tokenize(text)
    )
    return [word for (word, _) in fd.most_common(n)]


freq_words(constitution, 30)


# 4.4.6 文档说明函数(P163)
# Ex4-4: 完整的 docstring，
# 包括总结、详细的解释、doctest的例子以及特定的参数、类型、返回值类型和异常标记
def accuracy(reference, test):
    """
    Calculate the fraction of test items that equal the corresponding reference items.

    Given a list of reference values and a corresponding list of test values,
    return the fraction of corresponding values that are equal.
    In particular, return the fraction of indexes
    {0<i<=len(test)} such that C{test[i] == reference[i]}.

        >>> accuracy(['ADJ', 'N', 'V', 'N'], ['N', 'N', 'V', 'ADJ'])
        0.5

    :param reference: An ordered list of reference values
    :type reference: list
    :param test: A list of values to compare against the corresponding
        reference values
    :type test: list
    :return: the accuracy score
    :rtype: float
    :raises ValueError: If reference and length do not have the same length
    """

    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in zip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)
