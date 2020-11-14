# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0302.py
@Version    :   v0.1
@Time       :   2020-11-14 17:02
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


# 3.2. 最底层的文本处理：字符串处理
# 3.2.1 字符串的基本操作
# 字符串可以连接，列表只能追加，字符串和列表不能相互链接

# NLP 处理的流程：图 3-1
# 读入文件-->ASCII
# 标记文本-->创建NLTK文本
# 标准化处理文字-->创建词汇表


monty = 'Monty Python!'
print('monty= ', monty)

circus = "Monty Python's Flying Circus"
print('circus= ', circus)

circus = 'Monty Python\'s Flying Circus'
print('circus= ', circus)

couplet = "Shall I compare thee to a Summer's day?" \
          "Thou are more lovely and more temperate:"
print('couplet= ', couplet)

couplet = "Shall I compare thee to a Summer's day?" + \
          "Thou are more lovely and more temperate:"
print('couplet= ', couplet)

couplet = ("Shall I compare thee to a Summer's day?"
           "Thou are more lovely and more temperate:")
print('couplet= ', couplet)

couplet = """Shall I compare thee to a Summer's day?
Thou are more lovely and more temperate:"""
print('couplet= ', couplet)

couplet = '''Shall I compare thee to a Summer's day?
Thou are more lovely and more temperate:'''
print('couplet= ', couplet)

couplet = ("Rough winds do shake the darling buds of May,"
           "And Summer's lease hath all too short a date:")
print('couplet= ', couplet)

couplet = '''Rough winds do shake the darling buds of May,
And Summer's lease hath all too short a date:'''
print('couplet= ', couplet)

a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
print('a= ', a)

b = [' ' * 2 * (7 - i) + 'very' * i for i in a]
print('b= ', b)

# 3.2.2 字符串输出
print(monty)

grail = 'Holy Grail'
print(monty + grail)
print(monty, grail)
print(monty, 'and the', grail)

# 3.2.3 访问字符串中的单个字符
print("monty[0]= ", monty[0])
print("monty[3]= ", monty[3])
print("monty[5]= ", monty[5])
print("monty[len(monty) - 1]= ", monty[len(monty) - 1])

# 超出字符串的索引
# print("monty[20]= ", monty[20])

print("monty[-1]= ", monty[-1])
print("monty[-7]= ", monty[-7])

sent = 'colorless green ideas sleep furiously'
for char in sent:
    print(char, end=' ')

from nltk.corpus import gutenberg

raw = gutenberg.raw('melville-moby_dick.txt')
fdist = nltk.FreqDist(
        ch.lower()
        for ch in raw
        if ch.isalpha()
)
print("fdist.most_common(5)= ", fdist.most_common(5))

char_list = [
        char
        for (char, count) in fdist.most_common()
]
print('char_list= ', char_list)

# 3.2.4 访问子字符串（字符串切片）
print("monty[6:10]= ", monty[6:10])
print("monty[-12:-7]= ", monty[-12:-7])
print("monty[:5]= ", monty[:5])
print("monty[6:]= ", monty[6:])

phrase = 'And now for something completely different'
if 'thing' in phrase:
    print('found "thing"!')
print("monty.find('Python')= ", monty.find('Python'))

sent = "my sentence..."
print('sent slice= ', sent[3:11])

# 3.2.5 更多字符串操作函数：表3-2 P99

# 3.2.6 链表与字符串的差异
query = 'Who knows?'
print("query[2]= ", query[2])
print("query[:2]= ", query[:2])
print('query + " I don\'t"= ', query + " I don\'t")  # 字符串与字符串相加

# query[0] = 'F'  # 字符串不可以修改初始值
# del query[0]  # 字符串不可能删除里面的元素

beatles = ['John', 'Paul', 'George', 'Ringo']
print("beatles[2]= ", beatles[2])
print("beatles[:2]= ", beatles[:2])

# print("beatles + 'Brian'= ", beatles + 'Brian')  # 链表不能与字符串相加
print("beatles + ['Brian']= ", beatles + ['Brian'])  # 链表与链表相加

print("beatles= ", beatles)
beatles[0] = 'John Lennon'  # 链表可以修改初始值
print("beatles= ", beatles)
del beatles[0]  # 链表可以删除里面的元素
print("beatles= ", beatles)
