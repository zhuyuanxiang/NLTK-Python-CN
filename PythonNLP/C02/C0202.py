# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0202.py
@Version    :   v0.1
@Time       :   2020-11-12 15:00
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
from nltk.corpus import brown

from tools import *

# Chap2 获取文本语料库 和 词汇资源
# 目的：
# 1.  什么是有用的文本语料库和词汇资源？如何使用 Python 获取它们？
# 2.  哪些 Python 结构最适合这项工作？
# 3.  编写 Python 代码时如何避免重复的工作？

# 2.2. 条件频率分布：是频率分布的集合，每个频率分布有一个不同的“条件”。(condition,word)根据condition（条件）统计word（单词）的频率。
# 2.2.1. 条件 和 事件
# 2.2.2. 按文体计数词汇
cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in ['news', 'romance']
        for word in brown.words(categories=genre)
)
cfd.tabulate(samples=['the', 'cute', ' Monday', 'could', 'will'])

genre_word = [
        (genre, word)
        for genre in ['news', 'romance']
        for word in brown.words(categories=genre)
]
print("genre word count= ", len(genre_word))
print("genre_word[:4]= ", genre_word[:4])
print("genre_word[-4:]= ", genre_word[-4:])

cfd = nltk.ConditionalFreqDist(genre_word)
cfd
cfd.conditions()
print(cfd['news'])
print(cfd['romance'])
cfd['romance'].most_common(20)
cfd['romance']['could']

# 2.3. 绘制分布图 显示分布表
from nltk.corpus import inaugural

cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target)
)
from nltk.corpus import udhr

languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
        (lang, len(word))
        for lang in languages
        for word in udhr.words(lang + '-Latin1')
)
cfd.plot(cumulative=True)
cfd.tabulate()
cfd.tabulate(cumulative=True)
cfd.tabulate(samples=range(10), cumulative=True)
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)

from nltk.corpus import brown

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
brown_cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in ['news', 'romance']
        for word in brown.words(categories=genre)
        if word in days
)
brown_cfd.tabulate(samples=sorted(days))

# 2.2.4 使用二元语法（双边词）生成随机文本
# nltk.bigrams()生成连续的词对链表
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.for ']
list(nltk.bigrams(sent))

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

# 打印出来的结果 和 直接访问的结果不一样
print("bigrams= ", bigrams)
print("cfd['living']= ", cfd['living'])
print("dict(cfd['living'])= ", dict(cfd['living']))
print("cfd['living'].max()= ", cfd['living'].max())
print("cfd['finding'].max()= ", cfd['finding'].max())


# P59 Ex2-1 产生随机文本
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()  # 选择与原单词匹配度最大的单词作为下一个单词
    print()


show_subtitle('living')
generate_model(cfd, 'living')
show_subtitle('beginning')
generate_model(cfd, 'beginning')
show_subtitle('finding')
generate_model(cfd, 'finding')

generate_model(cfd, 'living')
generate_model(cfd, 'beginning')
generate_model(cfd, 'finding')
