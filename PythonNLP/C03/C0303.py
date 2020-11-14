# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0303.py
@Version    :   v0.1
@Time       :   2020-11-14 17:31
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
from tools import *

# Chap 3 处理原始文本
# 1）如何访问文件内的文本？
# 2）如何将文档分割成单独的单词和标点符号，从而进行文本语料上的分析？
# 3）如何产生格式化的输出，并把结果保存在文件中？

# 3.3. 使用 Unicode 进行文本处理
# 3.3.1 什么是Unicode？
# Unicode支持一百万种以上的字符。每个字符分配一个编号，称为编码点。
# 文件中的文本都有特定编码的，需要将文本翻译成Unicode，叫做解码。

# 3.3.2 从文件中提取已经编码的文本

# TypeError: expected str, bytes or os.PathLike object, not ZipFilePathPointer
# 报这个错误是因为文件不存在，并不是真的遇到错误的类型
# 报出这样的错误，可能是因为文件名不存在后，open再次调用一个空的path就会报错，需要重新初始化path再行
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')

f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line)

# unicode_escape 是一个虚拟的编码，将所有非 ASCII 字符转换成 \uXXXX 的形式
# 编码点在 ASCII 码 0~127 的范围以外但是低于 256 的，使用\xXX 的形式表示
f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

a = u'\u0061'
print("ord('a')= ", ord('a'))
print('a=', a)
print("a.encode('utf8')= ", a.encode('utf8'))

nacute = '\u0144'
print("ord('ń')= ", ord('ń'))
print('nacute= ', nacute)
print("nacute.encode('utf-8')= ", nacute.encode('utf-8'))
print("nacute.encode('utf8')= ", nacute.encode('utf8'))

# unicodedata 模块用于检查 Unicode 字符的属性
import unicodedata

lines = open(path, encoding='latin2').readlines()
line = lines[2]
line = line.lower()
print("line= ", line)
print("line.encode('unicode_escape')= ", line.encode('unicode_escape'))
print("line.encode('utf8')= ", line.encode('utf8'))
for c in line:
    if ord(c) > 127:
        print('{} = U+{:04x} = {} is {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c), c))

print("line.find('zosta\u0142y')= ", line.find('zosta\u0142y'))

line = line.lower()
print("line= ", line)
print("line.encode('unicode_escape')= ", line.encode('unicode_escape'))
print("line.encode('utf8')= ", line.encode('utf8'))
for c in line:
    if ord(c) > 127:
        print('{} = U+{:04x} = {} is {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c), c))

import re

m = re.search('\u015b\w*', line)
print("m.group()= ", m.group())
nltk.word_tokenize(line)

# 3.3.3 在 Python 中使用本地编码
# 需要在文件头加上字符串：# -*- coding: utf-8 -*-
import re

sent = """Przewiezione przez Niemcow pod knoiec II wojny swiatowej na Dolny Slask, 
zostaly odnalezione po 1945 r. na terytorium Polski."""

print("sent= ", sent)
bytes = sent.encode('utf8')
print("bytes= ", bytes)
print("bytes.lower()= ", bytes.lower())
print("bytes.decode('utf8')= ", bytes.decode('utf8'))

SACUTE = re.compile('s|S')
replaced = re.sub(SACUTE, '[sacute]', sent)
print("replaced= ", replaced)

# 字符 与 字符串 的转换
sent = "this is string example....wow!!!"
print("sent= ", sent)
bytes = sent.encode('utf8')
print("bytes= ", bytes)
# bytes.encode('utf8')  # bytes：字节不能再次编码
str = bytes.decode('utf8')
print("str= ", str)
# str.decode('utf8')  # str：字符串不能再次解码
