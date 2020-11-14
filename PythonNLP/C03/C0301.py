# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0301.py
@Version    :   v0.1
@Time       :   2020-11-14 8:51
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

# 3.1 从网络和硬盘访问文本
# 3.1.1 电子书
# [Gutenberg Electronic Books Project](http://www.gutenberg.org)
# 打不开就访问 http://www.gutenberg.org/ 搜索 2554
url = "http://www.gutenberg.org/files/25196/25196-0.txt"  # 编号 25196 的文本是《百家姓》

from urllib import request

response = request.urlopen(url)
raw = response.read().decode('utf8')
print("type(raw)= ", type(raw))
print("len(raw)= ", len(raw))
print("raw[:75]= ", raw[600:675])
# 网络比较慢，建议换下面的形式
from nltk.corpus import gutenberg

# gutenberg 默认的编码是 latin1
raw = gutenberg.raw('25196-0.txt').encode('latin1').decode('utf-8')

print("type(raw)= ", type(raw))
print("len(raw)= ", len(raw))
print("raw[:75]= ", raw[600:675])

# 分词：将字符串分解为词和标点符号
from nltk import word_tokenize

tokens = nltk.word_tokenize(raw)
print("type(tokens)= ", type(tokens))
print("len(tokens)= ", len(tokens))
print("tokens[:10]= ", tokens[:10])

# 文本切片
text = nltk.Text(tokens)
print("type(text)= ", type(text))
print("tokens[1020:1060]= ", tokens[1020:1060])
print("text[1020:1060]= ", text[1020:1060])

# 寻找文档中的固定搭配，不考虑停止词
text.collocations()

# 寻找文档中的关键信息（如：文本名称、作者名称、扫描和校对人名称、许可证信息等等）
print("raw.find(趙錢孫李')= ", raw.find('趙錢孫李'))
print("raw.rfind('周吳鄭王')= ", raw.rfind('周吳鄭王'))

# 3.1.2 处理HTML文件
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
url = "https://news.sina.com.cn/c/2019-06-22/doc-ihytcerk8630195.shtml"
url = "http://www.nltk.org/book/ch03.html"

html = request.urlopen(url).read()
print("type(html)= ", type(html))
print("html[:60]= ", html[:60])
html = html.decode('utf8')  # bytes 与 str 的转码
print("type(html)= ", type(html))
print("html[:60]= ", html[:60])

# 从 HTML 文件中获取文本
from bs4 import BeautifulSoup

raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw)
print("tokens[:20]= ", tokens[:20])
print("raw[:60]= ", raw[:60])

# 定位感兴趣的文本
tokens_part = tokens[96:399]
text = nltk.Text(tokens_part)
text.concordance('expressions')
text.concordance('Strings')
for i, word in enumerate(text):
    if i <= 13:
        print(word)

# 3.1.3 处理搜索引擎的结果。
# -   优点：很容易获得感兴趣的文本
# -   缺点：数据量大，并且随时会发现改变

# 3.1.4 处理 RSS 订阅
import feedparser

# http://languagelog.ldc.upenn.edu 打不开，估计被屏蔽了
# llog = feedparser.parse('http://languagelog.ldc.upenn.edu/nll/?feed=atom')
llog = feedparser.parse('http://www.gutenberg.org/cache/epub/feeds/today.rss')
print("llog['feed']['title']= ", llog['feed']['title'])
print("len(llog.entries)= ", len(llog.entries))

post = llog.entries[2]
print("post.title= ", post.title)

content = post.content[0].value
print("content[:70]= ", content[:70])

# 因为更换了 RSS，没有数据用于解析
raw = BeautifulSoup(content, 'html.parser').get_text()
print("word_tokenize(raw)= ", word_tokenize(raw))

# 3.1.5 读取本地文件
f = open('PythonNLP/dict.files/output.txt')
raw = f.read()
print("type(raw)= ", type(raw))
print("raw[100:110]= ",raw[100:110])

# 直接读取NLTK中的语料库文件
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'rU').read()
type(raw)

# 看看当前目录下面还有啥文件？
import os

os.listdir('..')

# 3.1.6 从 PDF、MS Word 及其他二进制格式中提取文本

# 3.1.7 捕获用户输入
sent = input('Enter somme text: ')
tokens = word_tokenize(sent)
type(tokens)
print('You typed', len(tokens), 'words.')
text = nltk.Text(tokens)
type(text)
words = [w.lower() for w in tokens]
type(words)
vocab = sorted(set(words))
type(vocab)

# 3.1.8 NLP 的流程
# NLP处理流程：
#   打开一个URL，读里面HTML 格式的内容，去除标记，并选择字符的切片，
#   然后分词，是否转换为nltk.Text 对象是可选择的。
#   也可以将所有词汇小写并提取词汇表。
# 注：中文需要提供分词功能
from bs4 import BeautifulSoup

url = "https://www.zhihu.com/"
html = request.urlopen(url).read().decode('utf8')
raw = BeautifulSoup(html).get_text()
raw = raw[:500]
tokens = word_tokenize(raw)
tokens = tokens[:390]
text = nltk.Text(tokens)
words = [w.lower() for w in text]
vocab = sorted(set(words))
show_subtitle("html[:20]")
print(html[:20])
show_subtitle("raw[:20]")
print(raw[:20])
show_subtitle("tokens[:20]")
print(tokens[:20])
show_subtitle("text[:20]")
print(text[:20])
show_subtitle("words[:20]")
print(words[:20])
show_subtitle("vocab[:20]")
print(vocab[:20])
