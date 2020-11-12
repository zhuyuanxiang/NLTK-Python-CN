# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0201.py
@Version    :   v0.1
@Time       :   2020-11-11 9:51
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
from nltk.corpus import gutenberg

from tools import *

# Chap2 获取文本语料库 和 词汇资源
# 目的：
# 1.  什么是有用的文本语料库和词汇资源？如何使用 Python 获取它们？
# 2.  哪些 Python 结构最适合这项工作？
# 3.  编写 Python 代码时如何避免重复的工作？

# 2.1. 获取文本语料库
# 2.1.1. 古腾堡语料库
show_subtitle("古腾堡语料库的文件列表")
print(nltk.corpus.gutenberg.fileids()[:5])

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print("len(emma)= ", len(emma))
show_subtitle("emma.concordance('surprise') ")
nltk.Text(emma).concordance('surprise')

# 通过 fileids() 函数遍历文件相关信息：平均词长、平均句子长度、文中每个单词出现的平均次数（词汇多样性得分）。文件名称
print("{0:4s}|{1:6s}|{2:7s}|{3:s}".format("平均词长", "平均句子长度", "词汇多样性得分", "文件名称"))
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print("{0:15d}|{1:21d}|{2:26d}|{3:s}".format(round(num_chars / num_words), round(num_words / num_sents),
                                                 round(num_words / num_vocab), fileid))

macbeth_sentences = gutenberg.sents("shakespeare-macbeth.txt")
print("macbeth_sentences= ", macbeth_sentences)
print("macbeth_sentences[1037]= ", macbeth_sentences[1037])

longest_len = max([len(s) for s in macbeth_sentences])
longest_sent = [s for s in macbeth_sentences if len(s) == longest_len]
print("longest_sent= ", longest_sent)

# 2.1.2. 网络文本 和 聊天文本
# 网络文本
from nltk.corpus import webtext

for field in webtext.fileids():
    print(field, webtext.raw(field)[:65], '...')

# 聊天文本
from nltk.corpus import nps_chat

for field in nps_chat.fileids():
    print(field, nps_chat.posts(field)[:12])

chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print("chatroom[123]= ", chatroom[123])

# 1.3. Brown（布朗）语料库：用于研究文体之间的系统性差异（又叫文体学研究）
from nltk.corpus import brown

show_subtitle("使用 categories 区分文本")
print("brown.categories() =", brown.categories())
print("brown.words(categories='news')= ", brown.words(categories='news'))
print("brown.words(categories=['news', 'editorial', 'reviews'])= ",
      brown.words(categories=['news', 'editorial', 'reviews']))
print("brown.sents(categories=['news', 'editorial', 'reviews'])= ",
      brown.sents(categories=['news', 'editorial', 'reviews']))

show_subtitle("使用 fileids 区分文本")
print("brown.words(fileids='cg22')= ", brown.words(fileids='cg22'))

news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=', ')

cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre)
)

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals)

wh_words = ['what', 'where', 'why', 'who', 'whom', 'how']
cfd.tabulate(conditions=genres, samples=wh_words)

show_subtitle("疑问词的使用计数")
for m in wh_words:
    print(m + ':', fdist[m], end=',')

# 2.1.4. 路透社语料库
from nltk.corpus import reuters

print("reuters.fileids()= ", reuters.fileids()[:12])
print("reuters.categories()= ", reuters.categories()[:12])

print(reuters.categories('training/9865'))
print(reuters.categories(['training/9865', 'training/9880']))
print(reuters.fileids('barley')[:12])
print(reuters.fileids(['barley', 'corn'])[:12])
print(reuters.words('training/9865')[:12])
print(reuters.words(['training/9865', 'training/9880'])[:12])
print(reuters.words(categories='barley'))
print(reuters.words(categories=['baryley', 'corn']))

# 2.1.5. 美国总统就职演说语料库
from nltk.corpus import inaugural

print(inaugural.fileids()[:12])
print([fileid[:4] for fileid in inaugural.fileids()])

# 统计演说词中 america 和 citizen 出现的次数
cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target)
)
# 图2-1：条件频率分布图
# 就职演说语料库中所有以 america 和 citizen 开始的词都将被计数。
# 每个演讲单独计数并绘制出图形，这样就能观察出随时间变化这些用法的演变趋势。
# 计数没有与文档长度进行归一化处理。
cfd.plot()

# 2.1.6. 标注文本语料库：表2-2 NLTK中的一些语料库和语料库样本
# 2.1.7. 其他语言的语料库
# 在Python 3.0中已经不存在字符编码问题
cess_esp_words = nltk.corpus.cess_esp.words()
print(cess_esp_words[:35])

floresta_words = nltk.corpus.floresta.words()
print(floresta_words[:35])

indian_words = nltk.corpus.indian.words()
print(indian_words[:35])

udhr_fileids = nltk.corpus.udhr.fileids()
print(udhr_fileids[:35])

udhr_words = nltk.corpus.udhr.words('Javanese-Latin1')
print(udhr_words[:35])

from nltk.corpus import udhr

languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
        (lang, len(word))
        for lang in languages
        for word in udhr.words(lang + '-Latin1')
)
# 图2-2：累积字长分布图
# 内容是「世界人权宣言」的 6 个翻译版本
# 此图显示： 5 个或者 5 个以下字母组成的词在 Ibiobio 语言的文本中占约 80%，在 German 中占约60%， 在 Inuktitut 中占约 25%
cfd.plot(cumulative=True)
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)

# 1.8. 文本语料库的结构
raw=gutenberg.raw('burgess-busterbrown.txt')
show_subtitle("文件中的原始内容")
print(raw[:123])
words=gutenberg.words('burgess-busterbrown.txt')
show_subtitle("文件中的单词")
print(words[:23])
sents=gutenberg.sents('burgess-busterbrown.txt')
show_subtitle("文件中的句子")
print(sents[:3])

chinese_mandarin_raw=udhr.raw('Chinese_Mandarin-UTF8')
show_subtitle("中文文件中的原始内容")
print(chinese_mandarin_raw[:13])

# 中文是字符型的，不能使用单词读入函数 words()
# chinese_mandarin_words=udhr.words('Chinese_Mandarin-UTF8')
# print(chinese_mandarin_words[:13])

# 中文是字符型的，不能使用句子读入函数 sents()
# chinese_mandarin_sents=udhr.sents('Chinese_Mandarin-UTF8')
# print(chinese_mandarin_sents[:13])

# 3.1.9. 载入自己的语料库
from nltk.corpus import PlaintextCorpusReader

# 这个在 C 盘根目录下，子目录中需要放入一些文件
corpus_root='/nltk_data/tokenizers/punkt'
word_lists=PlaintextCorpusReader(corpus_root,'.*')
print("自己语料库的文件列表= ", word_lists.fileids())

from nltk.corpus import BracketParseCorpusReader

corpus_root=r'C:\nltk_data\corpora\treebank\combined'
file_pattern=r'wsj_.*\.mrg'
ptb=BracketParseCorpusReader(corpus_root,file_pattern)

show_subtitle("文件列表")
print(ptb.fileids()[:13])

show_subtitle("句子列表")
print(ptb.sents()[:3])

show_subtitle("指定文件中的句子")
print(ptb.sents(fileids='wsj_0003.mrg')[19])