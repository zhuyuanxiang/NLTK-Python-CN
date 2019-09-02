import nltk

# Chap 3 处理原始文本
# 1）如何访问文件内的文本？
# 2）如何将文档分割成单独的单词和标点符号，从而进行文本语料上的分析？
# 3）如何产生格式化的输出，并把结果保存在文件中？


url = "http://www.gutenberg.org/files/2554/2554-0.txt"  # 这个也打不开了

from urllib import request

response = request.urlopen(url)
raw = response.read().decode('utf8')

from nltk.corpus import gutenberg

raw = gutenberg.raw('austen-emma.txt')

type(raw)
len(raw)
raw[:75]

# 分词：将字符串分解为词和标点符号
from nltk import word_tokenize

tokens = nltk.word_tokenize(raw)
type(tokens)
len(tokens)
tokens[:10]

# 文本切片
text = nltk.Text(tokens)
type(text)
text[1020:1060]

# 寻找文档中的固定搭配，不考虑停止词
text.collocations()

# 寻找文档中的关键信息（如：文本名称、作者名称、扫描和校对人名称、许可证信息等等）
raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")

# 处理HTML文件
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
url = "https://news.sina.com.cn/c/2019-06-22/doc-ihytcerk8630195.shtml"
url = "http://www.nltk.org/book/ch03.html"

html = request.urlopen(url).read()
type(html)
html = html.decode('utf8')  # bytes 与 str 的转码
type(html)
html[:60]

# 从 HTML 文件中获取文本
from bs4 import BeautifulSoup

raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw)
tokens

# 定位感兴趣的文本
tokens_part = tokens[96:399]
text = nltk.Text(tokens_part)
text.concordance('{')
text.concordance('频道')
for word in text:
    print(word)

# 处理搜索引擎的结果。
# 优点：很容易获得感兴趣的文本
# 缺点：数据量大，并且随时会发现改变

# 处理 RSS 订阅
import feedparser

llog = feedparser.parse('http://languagelog.ldc.upenn.edu/nll/?feed=atom')
llog['feed']['title']
len(llog.entries)
post = llog.entries[2]
post.title
content = post.content[0].value
content[:70]
raw = BeautifulSoup(content, 'html.parser').get_text()
word_tokenize(raw)

# 读取本地文件
f = open('output.txt')
raw = f.read()
type(raw)

# 直接读取NLTK中的语料库文件
path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
raw = open(path, 'r').read()
type(raw)

# 看看当前目录下面还有啥文件？
import os

os.listdir('.')

# 从 PDF、MS Word 及其他二进制格式中提取文本

# 捕获用户输入
sent = input('Enter somme text: ')
tokens = word_tokenize(sent)
type(tokens)
print('You typed', len(len(tokens)), 'words.')
text = nltk.Text(tokens)
type(text)
words = [w.lower() for w in tokens]
type(words)
vocab = sorted(set(words))
type(vocab)

# 3.2. 最底层的文本处理：字符串处理
# 字符串可以连接，列表只能追加，字符串和列表不能相互链接

# NLP 处理的流程：图 3-1
# 读入文件-->ASCII
# 标记文本-->创建NLTK文本
# 标准化处理文字-->创建词汇表


monty = 'Monty Python!'
circus = "Monty Python's Flying Circus"
circus = 'Monty Python\' s Flying Circus'

couplet = "Shall I compare thee to a Summer's day?" \
          "Thou are more lovely and more temperate:"
couplet = ("Rough winds do shake the darling buds of May,"
           "And Summer's lease hath all too short a date:")
couplet = """Shall I compare thee to a Summer's day?
Thou are more lovely and more temperate:"""
couplet = '''Rough winds do shake the darling buds of May,
And Summer's lease hath all too short a date:'''

a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
b = [' ' * 2 * (7 - i) + 'very' * i for i in a]

# 字符串输出
print(monty)

# 访问字符串中的单个字符
monty[0]
monty[len(monty) - 1]

# 访问子字符串（字符串切片）
monty[6:10]
monty[-12:-7]
monty[:5]
monty[6:]

phrase = 'And now for something completely different'
if 'thing' in phrase:
    print('found "thing"!')
monty.find('Python')

# 更多字符串操作函数：表3-2 P99

# 链表与字符串的差异
query = 'Who knows?'
beatles = ['John', 'Paul', 'George', 'Ringo']
query[2]
beatles[2]
query[:2]
beatles[:2]
query + " I don't"  # 字符串与字符串相加
beatles + 'Brian'
beatles + ['Brian']  # 链表与链表相加
beatles[0] = 'John Lennon'  # 链表可以修改初始值
query[0] = 'F'  # 字符串不可以修改初始值
del beatles[0]  # 链表可以删除里面的元素
del query[0]  # 字符串不可能删除里面的元素

# 3.3. 使用 Unicode 进行文本处理
# 什么是Unicode？ Unicode支持一百万种以上的字符。每个字符分配一个编号，称为编码点。
# 文件中的文本都有特定编码的，需要将文本翻译成Unicode，叫做解码。

# 从文件中提取已经编码的文本
# TypeError: expected str, bytes or os.PathLike object, not ZipFilePathPointer
# 报这个错误是因为文件不存在，并不是真的遇到错误的类型
# 报出这样的错误，可能是因为文件名不存在后，open再次调用一个空的path就会报错，需要重新初始化path再行
path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line)

f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

ord('a')
ord('ń')
nacute = '\u0144'
nacute.encode('utf-8')
nacute.encode('utf8')

import unicodedata

lines = open(path, encoding='latin2').readlines()
line = lines[2]
print(line)
print(line.encode('unicode_escape'))
print(line.encode('utf8'))

for c in line:
    if ord(c) > 127:
        print('{} U+{:04x} {} is {}'.format(c.encode('utf8'), ord(c), unicodedata.name(c), c))

line.find('zosta\u0142y')
line = line.lower()
line
line.encode('unicode_escape')
line.encode('utf8')

import re

m = re.search('\u015b\w*', line)
m.group()
word_tokenize(line)

# 在 Python 中使用本地编码时，需要在文件头加上字符串：# -*- coding: utf-8 -*-
import re

sent = """
Przewiezione przez Niemcow pod knoiec II wojny swiatowej na Dolny Slask, 
zostaly odnalezione po 1945 r. na terytorium Polski."""

bytes = sent.encode('utf8')
bytes.lower()
print(bytes.decode('utf8'))

SACUTE = re.compile('s|S')
replaced = re.sub(SACUTE, '[sacute]', sent)
print(replaced)

# 字符 与 字符串 的转换
sent = "this is string example....wow!!!"
bytes = sent.encode('utf8')
bytes.encode('utf8')  # bytes：字节不能再次编码
str = bytes.decode('utf8')
str.decode('utf8')  # str：字符串不能再次解码

# 3.4. 使用正则表达式检测词组搭配（本书可以帮助你快速了解）
# 取出所有的小写字母拼写的单词
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
# 搜索以'ed'结尾的单词
[w for w in wordlist if re.search('ed$', w)]
# 搜索以'**j**t**'形式的单词，'^'表示单词的开头，'$'表示单词的结尾
[w for w in wordlist if re.search('^..j..t..$', w)]
# [ghi]表示三个字母中任意一个
[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
# '+'表示一个或者多个
[w for w in chat_words if re.search('^m+i+n+e+$', w)]
# '*'表示零个或者多个
[w for w in chat_words if re.search('^m*i*n*e*$', w)]
# [^aeiouAEIOU]：表示没有这些字母的单词，即没有元音字母的单词，就是标点符号
[w for w in chat_words if re.search('[^aeiouAEIOU]', w)]

wsj = sorted(set(nltk.corpus.treebank.words()))
# 前面两个是一样的，因为小数肯定都会有整数在前面，而第三个不一样，是因为'?'表示零个或者一个，不包括大于10的整数
len([w for w in wsj if re.search('^[0-9]*\.[0-9]+$', w)])
len([w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)])
len([w for w in wsj if re.search('^[0-9]?\.[0-9]+$', w)])

[w for w in wsj if re.search('^[A-Z]+\$$', w)]
# 四位数的整数
[w for w in wsj if re.search('^[0-9]{4}$', w)]
[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]
[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]
# 寻找分词
[w for w in wsj if re.search('(ed|ing)$', w)]

# 表3-3 正则表达式的基本元字符，其中包括通配符、范围和闭包 P109
# 原始字符串（raw string）：给字符串加一个前缀“r”表明后面的字符串是原始字符串。

# 3.5. 正则表达式的有益应用
# P109 提取元音字符块
word = 'supercalifragilisticexpialidocious'
re.findall(r'[aeiou]', word)
len(re.findall(r'[aeiou]', word))

wsj = sorted(set(nltk.corpus.treebank.words()))
# P109 提取两个元音字符块
[vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word)]
# 统计双元音字符块的数目
fd = nltk.FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word))
fd.most_common(12)

# 提取日期格式中的整数值
[int(n) for n in re.findall(r'[0-9]+', '2009-12-31')]

# 使用findall()完成更加复杂的任务
# P110 忽略掉单词内部的元音
regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'  # 不明白为什么要设置这么复杂的模板？
regexp = r'[^AEIOUaeiou]'  # 这个简单的模板似乎也可以正常操作


def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)


compress('IiIiIi')
english_udhr = nltk.corpus.udhr.words('English-Latin1')
english_tmp = [compress(w) for w in english_udhr]
len(english_tmp)
len(''.join(english_tmp))
re.findall(r'[AEIOUaeiou]', ''.join(english_tmp))
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

# P111 提取所有辅音-元音序列对，并且统计单词库中这样的序列对的数目
rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[PTKSVRptksvr][AEIOUaeiou]', w.lower())]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

# 定义元音--辅音对应的单词集合
cv_word_pairs = [(cv, w) for w in rotokas_words for cv in re.findall(r'['r'ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
cv_index['su']
cv_index['po']


# P112 查找词干(忽略词语结尾，只处理词干）
def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word


# 只取出词尾（只提取了后缀，没有提出词干）
regexp = r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$'
re.findall(regexp, 'processing')

# 输出了整个单词（提取符合后缀的字符串，"(?:)"的作用，但是没有提取出词干）
regexp = r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$'
re.findall(regexp, 'processing')  # 符合词缀要求的单词可以提取出来
re.findall(regexp, 'processooo')  # 不符合词缀要求的单词就不提取出来

# 将单词分解为词干和后缀
regexp = r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$'
re.findall(regexp, 'processing')
re.findall(regexp, 'processes')  # 使用贪婪匹配模式，错误分解单词

# 不使用贪婪匹配模式
regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$'
re.findall(regexp, 'processes')
re.findall(regexp, 'process')  # 需要单词背景知识，将这类单词剔除，否则会错误地提取词干
re.findall(regexp, 'language')  # 没有单词背景知识时，如果对于没有词缀的单词会无法提取出单词来

# 正确处理没有后缀的单词
regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
re.findall(regexp, 'language')


# 更加准确地词干提取模板，先将原始数据分词，然后提取分词后的词干
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem


raw = """DENNIS: Listen,
strange women lying in ponds distributing swords
is no basis for a system of government.
Supreme executive power derives from a mandate from the masses, 
not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)
[stem(t) for t in tokens]

# 正则表达式的展示函数，可以把符合正则表达式要求的字符标注出来
regexp = r'[ing|ly|ed|ious|ies|ive|es|s|ment]$'
regexp = r'(ing)$'
regexp = r'[ing]$'
regexp = r'ing$'

nltk.re_show(regexp, raw)  # 不能使用re.findall()中的正则表达式标准。需要使用基本的正则表达式标准。
# P109 表3-3 正则表达式基本元字符，P120 表3-4 正则表达式符号
# 也可以参考《自然语言处理综论（第二版）》P18
nltk.re_show('^[D|s|i|S|n]', raw)  # '^' 表示行的开头
nltk.re_show('^[DsiSn]', raw)  # '[]' 内，用不用|都表示析取
nltk.re_show('[s|.|,]$', raw)  # '$' 表示行的结尾
nltk.re_show('ing|tive', raw)  # '|' 表示析取指定的字符串
nltk.re_show('(ing|tive)', raw)  # '()' 表示操作符的范围
nltk.re_show('(s){1,2}', raw)  # '{}' 表示重复的次数

# P114 对已经实现分词的文本（Text）进行搜索（findall）
from nltk.corpus import gutenberg, nps_chat

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
tokens = moby.tokens

regexp = r"(?:<a> <.*> <man>)"
regexp = r"<a>(<.*>)<man>"
moby.findall(regexp)

nltk.re_show('ly|ed|ing', ' '.join(tokens[:75]))
# 找出文本中"a <word> man"中的word
nltk.re_show('see [a-z]+ now', ' '.join(tokens[:200]))

chat = nltk.Text(nps_chat.words())
tokens = chat.tokens

regexp = r"<.*><.*><bro>"
regexp = r"<l.*>{3,}"
chat.findall(regexp)
# nltk.re_show() 本身对正则表达式的支持就不完全按标准来
nltk.re_show('l.+', ' '.join(tokens[:200]))
nltk.re_show('h.+', ' '.join(tokens[200:1000]))

# 正则表达式的测试界面
nltk.app.nemo()

from nltk.corpus import brown

hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>")

# P115 3.6. 规范化文本
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  
 Supreme executive power derives from a mandate from the masses, 
 not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)

# 词干提取器 Porter 比 Lancaster 要好点
# Porter 词干提取器
porter = nltk.PorterStemmer()
[porter.stem(t) for t in tokens]

# Lancaster 词干提取器
lancaster = nltk.LancasterStemmer()
[lancaster.stem(t) for t in tokens]


# Ex3-1 使用词干提取器索引文本
class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width / 4)
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i - wc:i])
            rcontext = ' '.join(self._text[i:i + wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()


grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('em')

# P117 词形归并，WordNet词形归并器将会删除词缀产生的词，即将变化的单词恢复原始形式，例如：women转变成woman

wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

# P118 3.7. 用正则表达式为文本(Text)分词，分词（Tokenization）


raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful 
tone though), 'I won't have any pepper in my kitchen AT ALL. Soup does very 
well without--Maybe it's always pepper that makes people hot-tempered,'..."""

re.split(r' ', raw)  # 利用空格分词，没有去除'\t'和'\n'
re.split(r'[ \t\n]+', raw)  # 利用空格、'\t'和'\n'分词，但是不能去除标点符号
re.split(r'\s', raw)  # 使用re库内置的'\s'（匹配所有空白字符）分词，但是不能去除标点符号
re.split(r'\W+', raw)  # 利用所有字母、数字和下划线以外的字符来分词，但是将“I'm”、“won't”这样的单词拆分了
re.findall(r'\w+|\S\w*', raw)  # 使用findall()分词，可以将标点保留，不会出现空字符串
re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw)  # 利用规则使分词更加准确

# P120 NLTK 的正则表达式分词器

text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
  | \w+(-\w+)*        # words with optional internal hyphens
  | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
  | \.\.\.            # ellipsis
  | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''

# TODO: 无法得出书上的结果
# “(?x)”为pattern中的“verbose”标志，将pattern中的空白字符和注释都去掉。
nltk.regexp_tokenize(text, '([A-Z]\.)')
nltk.regexp_tokenize(text, '\w+')  # TODO：不知道为什么这个与下面那个都理论上一样，显示结果不一样
nltk.regexp_tokenize(text, '\w(\w)+')
nltk.regexp_tokenize(text, '\w+()*')  # 可能问题还是在我自己的理解上
nltk.regexp_tokenize(text, '\w+(-\w+)')
nltk.regexp_tokenize(text, '\w+(-\w+)*')
nltk.regexp_tokenize(text, '\.\.\.|([A-Z]\.)+')
nltk.regexp_tokenize(text, '(?x)([A-Z]\.)+|\w+(-\w+)*|\.\.\.')

# P121 3.8 分割 Segmentation

# P122 句分割，断句，Sentence Segmentation
# 计算布朗语料库中每个句子的平均词数
len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)  # 转为使用Punkt句子分割器
sents[171:181]


# P123 Ex3-2 词分割，分词，Word Segmentation
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i + 1])
            last = i + 1
    words.append(text[last:])
    return words


text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"

seg1 = "0000000000000001000000000010000000000000000100000000000"
segment(text, seg1)

seg2 = "0100100100100001001001000010100100010010000100010010000"
segment(text, seg2)

words = segment(text, seg2)
text_size = len(words)
lexicon_size = len(' '.join(list(set(words))))
text_size + lexicon_size


# P124 Ex3-3 计算存储词典和重构源文本的成本，计算目标函数，评价分词质量，得分越小越好
def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size


text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"

seg1 = "0000000000000001000000000010000000000000000100000000000"
evaluate(text, seg1)

seg2 = "0100100100100001001001000010100100010010000100010010000"
evaluate(text, seg2)

seg3 = "0000100100000011001000000110000100010000001100010000001"
evaluate(text, seg3)

# P125 Ex3-4 使用模拟退火算法的非确定性搜索；
# 1) 一开始仅搜索短语分词；
# 2) 然后随机扰动0和1，它们与“温度”成比例；
# 3) 每次迭代温度都会降低，扰动边界会减少。

from random import randint


def flip(segs, pos):
    return segs[:pos] + str(1 - int(segs[pos])) + segs[pos + 1:]


def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs) - 1))
    return segs


def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, int(round(temperature)))
            score = evaluate(text, guess)
            if score < best:
                best_segs, best = guess, score
        segs, score = best_segs, best
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs


anneal(text, seg1, 5000, 1.2)
anneal(text, seg2, 5000, 1.2)

# P126 3.9 格式化：从链表到字符串
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
output=' ,'.join(['{} ({})'.format(word,len(word)) for word in saying])
wrapped = fill(output)  # 自动换行显示
print(wrapped)
