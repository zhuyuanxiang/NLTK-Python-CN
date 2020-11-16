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

from tools import show_subtitle

# 3.9 格式化：从链表到字符串(P126)
# 3.9.1 从链表转换为字符串
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
' '.join(silly)

# 3.9.2 字符串显示方式（两种）
word = 'cat'
print(word)
print(word.encode('utf-8'))

# print()函数按文本输出的格式输出，sentence或者 sentence.encode()则按字符串具体的内容输出
sentence = """hello 
world"""
print(sentence)  # 以可读的形式输出对象的内容
print(sentence.encode('utf-8'))  # 变量提示

fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
fdist.tabulate()
# 三种格式化输出文本的方法
for word in sorted(fdist):
    print(word, '->', fdist[word], end=':')
    print('{}->{};'.format(word, fdist[word]), end=' ')  # fromat()函数格式化输出文本
    print('%s->%d,' % (word, fdist[word]), end=' ')
    print('{1}->{0}'.format(fdist[word], word))

template = 'Lee wants a {} right now.'
menu = ['sandwich', 'spam fritter', 'pancake']
for snack in menu:
    print(template.format(snack))

# 3.9.3 排列
# 将文本按列排版
print("左边靠齐，6个字符=> |{:6}{:6}{:6}".format('dog', 'cat', 'man'))
print("右边靠齐，6个字符=> |{:>6}{:>6}{:>6}".format('dog', 'cat', 'man'))

import math

# 浮点数，小数点后4位
print('{:.4f}'.format(math.pi))

count, total = 3205, 9375
# 百分数，小数点后4位
print('accuracy for {} words: {:.4%}'.format(total, count / total))


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

cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre)
)

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)
print("cfd['news']['book']= ", cfd['news']['book'])

# 通过使用变量指定字段的宽度
print('{:{width}}'.format('Monty Python', width=15) + '!')
print(''.join([str(i) for i in range(10)]) * 2)

# 3.9.4 将结果写入文件(P130)
# 输出文件的两种方式：print(str,file=output_file), output_file.write(str)
# print()输出时默认在行结束时加入了换行符
output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    print(word, file=output_file)
print(str(len(words)), file=output_file)
output_file.write('zYx.Tom')  # 返回写入的字符个数
output_file.write(str(len(words)) + '\n')  # 没有'\n'则会连续写，不换行
output_file.flush()  # 刷新写文件缓冲区
output_file.close()

# 3.9.5 文本换行(Text Wrapping)(P131)
# 文本过长，到行尾溢出
saying = ['After', 'all', 'is', 'said', 'and', 'done', ',', 'more', 'is', 'said', 'than', 'done', '.']

for word in saying:
    print(word, '(' + str(len(word)) + ')', end=' ')

# 文本显示时自动换行
from textwrap import fill

format = '%s_(%d)'
pieces = [format % (word, len(word)) for word in saying]
output = ', '.join(pieces)
wrapped = fill(output)  # 自动换行显示
show_subtitle(format)
print(wrapped)

format = '{}_({})'
pieces = [f'{word}_({len(word)})' for word in saying]
output = ', '.join(pieces)
wrapped = fill(output)  # 自动换行显示
show_subtitle(format)
print(wrapped)

# 3.10 小结
# -   字符串中的字符是使用索引来访问的，索引从零开始计数(`str[0]`)
# -   子字符串使用切片符号访问(`str[3:5]`)
# -   字符串可以被分割成链表(`str.split()`);链表还可以连接成字符串`''.join(list)`。
# -   文本可以从文件中读取，也可以从URL地址中读取。
# -   分词是将文本分割成基本单位或者标记，例如：词和标点符号等。基于空格符的分词无法满足应用需要。
# -   词形归并是一个过程，将一个词的各种形式遇到这个词的标准形式或者引用形式，也称为词位或者词元。
# -   正则表达式是用来指定模式的方法，re.findall() 可以找到一个字符串中匹配一个模式的所有子字符串。
# -   在正则字符串前加上前缀`r`，提醒 Python 这个是正则表达式的字符串，不要处理包含的反斜杠。
# -   字符串格式化表达式包含格式字符串及转换标识符。
