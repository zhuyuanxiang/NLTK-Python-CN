# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0308.py
@Version    :   v0.1
@Time       :   2020-11-16 10:18
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
import nltk

# 3.9 格式化：从链表到字符串(P126)
# 如何把链表转换为字符串
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
' '.join(silly)

# 字符串显示方式（两种）
word = 'cat'
# print()函数按文本输出的格式输出，sentence则按字符串具体的内容输出
sentence = """hello 
world"""
print(sentence)  # 以可读的形式输出对象的内容
sentence  # 变量提示

fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
fdist.tabulate()
# 三种格式化输出文本的方法
for word in sorted(fdist):
    print(word, '->', fdist[word], end=':')
    print('{}->{};'.format(word, fdist[word]), end=' ')  # fromat()函数格式化输出文本
    print('%s->%d,' % (word, fdist[word]), end=' ')
    print('{1}->{0}'.format(fdist[word], word))

'from {1} to {0}'.format('A', 'B')

template = 'Lee wants a {} right now.'
menu = ['sandwich', 'spam fritter', 'pancake']
for snack in menu:
    print(template.format(snack))

# 将文本按列排版
'{:6}'.format('dog')  # 左边靠齐，6个字符
'{:>6}'.format('dog')  # 右边靠齐，6个字符

import math

'{:.4f}'.format(math.pi)  # 浮点数，小数点后4位

count, total = 3205, 9375
'accuracy for {} words: {:.4%}'.format(total, count / total)  # 百分数，小数点后4位


# Ex3-5 布朗语料库中情态动词在不同类别中的频率统计
def tabulate(cfdist, words, categories):
    print('{:16}'.format('Category'), end=' ')
    for word in words:  # 不同情态动词的题头
        print('{:>6}'.format(word), end=' ')
    print()
    for category in categories:  # 不同类别
        print('{:16}'.format(category), end=' ')
        for word in words:  # 不同情态动词
            print('{:6}'.format(cfdist[category][word]), end=' ')
        print()


from nltk.corpus import brown

cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories()
                               for word in brown.words(categories=genre))

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)
cfd['news']['book']

'{:{width}}'.format('Monty Python', width=15) + '!'

# P130 将结果写入文件
output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    print(word, file=output_file)
print(str(len(words)), file=output_file)
output_file.write('zYx.Tom')  # 返回写入的字符个数
output_file.write(str(len(words)) + '\n')  # 没有'\n'则会连续写，不换行
output_file.flush()  # 刷新写文件缓冲区

# P131 文本换行, Text Wrapping （文本显示时自动换行）
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more', 'is', 'said', 'than', 'done', '.']

for word in saying:
    print(word, '(' + str(len(word)) + ')', end=' ')

from textwrap import fill

format = '%s (%d),'
pieces = [format % (word, len(word)) for word in saying]
output = ' '.join(pieces)
output = ' ,'.join(['{} ({})'.format(word, len(word)) for word in saying])
wrapped = fill(output)  # 自动换行显示
print(wrapped)
