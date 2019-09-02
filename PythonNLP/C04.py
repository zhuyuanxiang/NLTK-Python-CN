import nltk

# Chap4 编写结构化的程序
# 4.1. 回到Python语言的基础
foo = 'Monty'
bar = foo
foo = 'Python'
bar

foo = ['Monty', 'Python']
bar = foo  # bar 对 foo 只是引用，所以 foo 的改变也会影响到 bar 的值
foo[1] = 'Bodkin'
foo

empty = []
nested = [empty, empty, empty]
nested[1].append('Python')
nested

nested = [[]] * 3
nested[1].append('Python')
nested
nested[1] = ['Monty']
nested
nested[1].append('Monty Python')
nested
nested[0].append('Little Python')
nested

# 复制链表（三种拷贝）
import copy

# 1:直接复制（用等于号），只复制引用

foo = ['Monty', 'Python', [1, 2]]
bar = foo
bar == foo
foo[1] = 'Bodkin'
foo[2][0] = 3
foo
bar

# 2: shadow copy浅拷贝，只复制浅层结构和深层次的引用
foo = ['Monty', 'Python', [1, 2]]
bar = copy.copy(foo)
bar == foo
foo[1] = 'Bodkin'
foo[2][0] = 3
foo
bar

# 3: deep copy 深拷贝，不复制任何引用，只复制结构
foo = ['Monty', 'Python', [1, 2]]
bar = copy.deepcopy(foo)  # 复制 foo 的结构，而不复制引用
bar == foo
foo[1] = 'Bodkin'
foo[2][0] = 3
foo
bar

# 等式
size = 5
python = ['Python']
snake_nest = [python] * size
snake_nest.insert(0, ['Python'])
snake_nest
snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]
snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4] is snake_nest[0]
snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]

import random

size = 5
position = random.choice(range(size))
snake_nest[position] = ['Python']
snake_nest
snake_nest[0] == snake_nest[1] == snake_nest[2] == snake_nest[3] == snake_nest[4]
snake_nest[0] is snake_nest[1] is snake_nest[2] is snake_nest[3] is snake_nest[4]
snake_nest.insert(0, ['Monty Python'])
snake_nest

# id(snake) 字符串签名编码
[id(snake) for snake in snake_nest]

# 条件判别语句
mixed = ['cat', '', ['dog'], []]
for element in mixed:
    if element:  # 非空字符串或非空链表判为真；空字符串或空链表判为假
        print(element)

# 判决短路，第一个条件满足以后，后面的就不再判别就不再执行
animals = ['cat', 'dog']
if 'cat' in animals:
    print(1)
elif 'dog' in animals:
    print(2)

sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
all(len(w) > 4 for w in sent)  # all()检查全部满足条件
any(len(w) > 4 for w in sent)  # any()检查部分满足条件

# P147 4.2 序列（Sequences）：序列对象共有三种（字符串、链表、元组)
# 元组的定义与使用
t1 = 'walk', 'fem', 3  # 定义元组的标准方法
t2 = ('walk', 'fem', 3)  # 括号并非定义元组的标准方法，括号是Python语法的一般功能，是用于分组的
t1 == t2
t1[0]
t2[1:]
len(t1)
t3 = ['a', 'b', 'c'], 'a', 'b', 'c'
t4 = (['a', 'b', 'c'], 'a', 'b', 'c')
t3 == t4
emtpy_tuple = ()  # 空元组的定义
single_tuple = 'love',  # 单个元素的元组的定义

# spectroroute这个单词查不到
raw = 'I turned off the spectroroute'  # 字符串
text = ['I', 'turned', 'off', 'the', 'spectroroute']  # 链表
pair = (6, 'turned')  # 元组
print('str:{}\nlist:{}\ntuple:{}'.format(raw[2], text[3], pair[1]))
print('str:{}\nlist:{}\ntuple:{}'.format(raw[-3:], text[-3:], pair[-3:]))
print('str:{}\nlist:{}\ntuple:{}'.format(len(raw), len(text), len(pair)))

# 定义集合
# 集合不可以索引访问
sets = set(raw)
len(sets)
'f' in sets
lists = list(sets)
len(lists)
'f' in lists

sets = set(text)
sorted(sets)

# P148 序列类型上的操作
raw = 'Red lorry, yellow lorry, red lorry, yellow lorry'
set(raw)  # 不能出现重复的元素
list(raw)  # 可以出现重复的元素
from nltk import word_tokenize

text = word_tokenize(raw)  # 分词
fdist = nltk.FreqDist(text)  # 词频统计
sorted(fdist)
for key in fdist:
    print(key + ':', fdist[key])

raw = 'I turned off the spectroroute'
text = word_tokenize(raw)
text
words = ['I', 'turned', 'off', 'the', 'spectroroute']
# 利用元组批量赋值
words[2], words[3], words[4] = words[3], words[4], words[2]
words

# zip() 取出两个或者两个以上的序列中的项目，将它们“压缩”打包成单个的配对链表
# 数目不匹配的部分直接丢弃，例如：words比赛tags多一个，就直接丢弃了。
words = ['I', 'turned', 'off', 'the', 'spectroroute', '!']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
list(zip(words, tags))
list(zip(words, tags, tags))

# enumerate() 返回一个包含索引及索引处所在项目的配对。
list(enumerate(words))
for a, b in enumerate(words):
    print('a:{}\tb:{}'.format(a, b))

# 分割数据集为（训练集+测试集)
text = nltk.corpus.nps_chat.words()
cut = int(0.9 * len(text))
training_data, test_data = text[:cut], text[cut:]
len(training_data) / len(test_data)

# 使用split()函数分词
raw = 'I turned off the spectroroute'
words = raw.split()
wordlens = [(len(word), word) for word in words]
wordlens.sort()
wordlens.reverse()
wordlens.pop()
' '.join(w for (_, w) in wordlens)

# 元组是不可修改的，而链表是可以修改的
lexicon = [('the', 'det', ['Di:', 'D@']), ('off', 'prep', ['Qf', 'O:f'])]
lexicon.sort()
lexicon
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
lexicon
del lexicon[0]
lexicon
# 链表转换成元组后，下面的操作都不可执行
lexicon = tuple(lexicon)
lexicon.sort()
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3`nd'])
del lexicon[0]
# 使用元组还是链表主要还是取决于项目的内容是否与它的位置相关

# 产生器表达式（使用列表推导式方便处理文本)
text = '''"When I use a word," Humpty Dumpty said in rather a scornful tone, 
"it means just what I choose it to mean - neither more nor less."'''
text
words = [w.lower() for w in nltk.word_tokenize(text)]
words[0] == words[7]  # 句子开头和结尾的双引号是有区别的
# 这两种方式都会先生成链表对象（占存储空间)
max(words)
max([w.lower() for w in nltk.word_tokenize(text)])
# max()函数调用时不仅省略了方括号[]，还利用数据流向调用它的函数的方式，避免存储过多的数据
max(w.lower() for w in nltk.word_tokenize(text))

# P152 4.3 风格的问题
# Python代码的风格
import re
from nltk.corpus import brown

rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cv_words_pairs = [(cv, w) for w in rotokas_words
                  for cv in re.findall('[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories()
                               for word in brown.words(categories=genre))
ha_words = ['aaahhhh', 'ah', 'ahah', 'ahahah', 'ahh', 'ahhahahaha',
            'ahhh', 'ahhhh', 'ahhhhhh', 'ahhhhhhhh', 'ha', 'haaa', 'hah', 'haha', 'hahaaa', 'hahah', 'hahaha']

syllables = []


def process():
    return


if (len(syllables) > 4 and len(syllables[2]) == 3 and
        syllables[2][2] in [aeiou] and syllables[2][3] == syllables[1][3]):
    process(syllables)

# 过程声明的风格
# 统计布朗语料库中词的平均长度
tokens = nltk.corpus.brown.words(categories='news')
count = 0
total = 0
for token in tokens:
    count += 1
    total += len(token)
print('total / count={:.3f}'.format(total / count))

[len(t) for t in tokens]
total = sum(len(t) for t in tokens)
print('total / count={:.3f}'.format(total / len(tokens)))

word_list = sorted(set(tokens[:75]))
# 下面是等效的代码，但是速度慢得出奇
word_list = []
i = 0
while i < len(tokens[:75]):
    j = 0
    while j < len(word_list) and word_list[j] <= tokens[i]:
        j += 1
    if j == 0 or tokens[i] != word_list[j - 1]:
        word_list.insert(j, tokens[i])
    i += 1
print(word_list)

# 统计布朗语料库中单词占比数，超过25%后停止输出
fd = nltk.FreqDist(nltk.corpus.brown.words())
cumulative = 0.0
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    cumulative += fd.freq(word)  # word在总文本中的占比数
    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
    if cumulative > 0.25:
        break

# P155 寻找最长的单词
text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
    if len(word) > len(longest):
        longest = word
print('longest word:{}'.format(longest))

# 下面是等效的代码，使用两个链表推导式
maxlen = max(len(word) for word in text)
[word for word in text if len(word) == maxlen]

# 计数器（counter）的常规用法
# 使用手环变量来提取链表中连续重叠的n-grams
n = 3
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
[sent[i:i + n] for i in range(len(sent) - n + 1)]
# 下面是等效的代码
list(nltk.bigrams(sent))
list(nltk.trigrams(sent))
list(nltk.ngrams(sent, 4))

# 使用循环变量构建多维结构
m, n = 3, 7
array = [[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
import pprint

pprint.pprint(array)

array = [[set()] * n] * m
array[2][5].add(7)
array[2][5].add(8)
array[2][5].add(9)
pprint.pprint(array)

# P156 4.4 结构化编程的基础
# 从文件中读取文本
import re


def get_text(file):
    """Read text from a file, normalizing whitespace and stripping HTML markup."""
    text = open(file).read()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('\s+', '', text)
    return text


help(get_text)
contents = get_text('output.txt')


# 函数的输入与输出
def repeat(msg, num):
    return ' '.join([msg] * num)


monty = 'Monty Python'
repeat(monty, 3)


def monty():
    return 'Monty Python'


monty()

repeat(monty(), 3)
repeat('Monty Python', 3)


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


# P159 参数传递
# 第一个传入参数在函数内部被改变了，但是调用者的参数内容没有被改变，因为是按内容传值的
# 第二个传入参数在函数内部被改变了，调用的参数内容也被改变了，因为是按地址传值的
def set_up(word, properties):
    word = 'lolcat'
    properties.append('noun')
    properties = 5


# w没有被改变，p被改变了
w = ''
p = []
set_up(w, p)
w
p

# w没有被改变
w = ''
word = w
word = 'lolcat'
w

# p被改变了
p = []
properties = p
properties.append('noun')
properties = 5
p
properties


# 变量的作用域：名称解析的LGB规则（本地（local）、全局（global）、内置（built-in））

# P160 参数类型检查
# 没有参数类型检查的函数
def tag(word):
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'


tag('the')
tag('knight')
# 传入链表后，函数返回值是错误的
tag(["'Tis", 'but', 'a', 'scratch'])


def tag(word):
    assert isinstance(word, str), "argument to tag() must be a string"
    if word in ['a', 'the', 'all']:
        return 'det'
    else:
        return 'noun'


tag('the')
tag('knight')
# 传入链表后，函数断言失败
tag(["'Tis", 'but', 'a', 'scratch'])


# 功能分解
def load_corpus():
    return


def analyze(data):
    return


def present(results):
    return


data = load_corpus()
results = analyze(data)
present(results)

from urllib import request
from bs4 import BeautifulSoup


# Ex4-2 计算高频词的拙劣函数，存在的几个问题：
# 1）修改了第二个参数的内容
# 2）输出了已经计算过的结果
def freq_words(url, freqdist, n):
    """

    :param url:
    :type url:
    :param freqdist:
    :type freqdist:
    :param n:
    :type n:
    """
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html, 'html.parser').get_text()
    for word in word_tokenize(text):
        freqdist[word.lower()] += 1
    result = []
    for word, count in freqdist.most_common(n):
        result += [word]
    print(result)


consitution = "http://www.archives.gov/exhibits/charters/constitution_transcript.html"
fd = nltk.FreqDist()
freq_words(consitution, fd, 30)


# 重构Ex4-2该函数，得到Ex4-3 用来计算高频词的函数
def freq_words(url, n):
    html = request.urlopen(url).read().decode('utf8')
    text = BeautifulSoup(html, 'html.parser').get_text()
    freqdist = nltk.FreqDist(word.lower() for word in word_tokenize(text))
    return [word for (word, _) in fd.most_common(n)]


freq_words(consitution, 30)


# P163 文档说明函数 docstring
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


# P164 4.5 更多关于函数
# 作为参数的函数
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the', 'sounds', 'will', 'take', 'care', 'of', 'themselves',
        '.']


def extract_property(prop):
    return [prop(word) for word in sent]


# 将函数作为参数传入，并且在函数内调用
extract_property(len)


def last_letter(word):
    return word[-1]


# 将自定义函数作为参数传入，并且在函数内调用
extract_property(last_letter)
# 将lambda表达式作为参数传入，并且在函数内调用
extract_property(lambda w: w[-1])

# 将函数作为参数传给sorted()函数
# 这个版本的Python的sorted()函数不再支持函数传入
sorted(sent)


# 累积函数
# Ex4-5 累积输出到一个链表
def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result


# 生成器函数的累积输出
# 函数只产生调用程序需要的数据，并不需要分配额外的内存来存储输出
def search2(substring, words):
    for word in words:
        if substring in word:
            yield word


# 全部搜索完毕再输出
print("search1:")
for item in search1('zz', nltk.corpus.brown.words()):
    print(item)

# 按照次序找到一个就输出一个
print("search2:")
i = 0
for item in search2('zz', nltk.corpus.brown.words()):
    print('{})'.format(i), item)
    i += 1


# 更复杂的生成器的例子
def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]


list(permutations(['police']))
list(permutations(['police', 'fish']))
list(permutations(['police', 'fish', 'buffalo']))
list(permutations(['police', 'fish', 'buffalo', 'cowboy']))
list(permutations(['police', 'fish', 'buffalo', 'cowboy', 'gun']))
list(permutations(['police', 'fish', 'buffalo', 'cowboy', 'gun', 'field']))

# 高阶函数，函数式编程，Haskell
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the', 'sounds', 'will', 'take', 'care', 'of', 'themselves',
        '.']


# 检查一个词是否来自一个开放的实词类
def is_content_word(word):
    return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']


# 将is_content_word()函数作为参数传入filter()函数中，就是高阶函数
list(filter(is_content_word, sent))
# 下面是等价的代码
[w for w in sent if is_content_word(w)]

# 将len()函数作为参数传入 map()函数中，就是高阶函数
lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))
# list()函数不支持将函数作为参数传入
list(len, ['a', 'b', 'c'])
sum(lengths) / len(lengths)  # 布朗语料库中句子的平均长度
lengths = [len(w) for w in nltk.corpus.brown.sents(categories='news')]
sum(lengths) / len(lengths)
# 多重函数嵌套
len(list(filter(lambda c: c.lower() in "aeiou", 'their')))
list(map(lambda w: len(list(filter(lambda c: c.lower() in "aeiou", w))), sent))
# 下面是等价的代码，以链表推导为基础的解决方案通常比基于高阶函数的解决方案可读性更好
[len([c for c in w if c.lower() in "aeiou"]) for w in sent]

[[c for c in w if c.lower() in "aeiou"] for w in sent]

[c for w in sent for c in w if c.lower() in "aeiou"]


# P167 参数的命名
# 关键字参数：就是通过名字引用参数，可以防止参数混淆，还可以指定默认值，还可以按做生意顺序访问参数
def repeat(msg='<empty>', num=1):
    return msg * num


repeat(num=3)
repeat(msg='Alice')
repeat(num=3, msg='Alice')

# 参数元组：*args 和 关键字参数字典：*kwargs
dict = {'a': 1, 'b': 2, 'c': 3}
list(dict)
dict.keys()
dict.values()
dict.items()
dict['a']
dict.get('a')


def generic(*args, **kwargs):
    for value in args:
        print('value={}'.format(value))
    print(kwargs['monty'])
    for name in list(kwargs):
        print('name:value={}: {}'.format(name, kwargs[name]))


generic(1, "African swallow", monty='Python')

# zip() 函数对 *args 的支持
song = [['four', 'calling', 'birds'], ['three', 'French', 'hens'], ['two', 'turtle', 'doves'],
        ['one', 'tiger', 'animal']]
list(zip(song[0], song[1], song[2]))
list(zip(*song))  # *song 表示分解输入数据，等价于song[0],song[1],song[2],song[3]，分组的数目就是输入的数目
list(zip(song))


# 三种等效的方法调用函数 freq_words()
def freq_words(file, min=1, num=10):
    text = open(file).read()
    tokens = nltk.word_tokenize(text)
    freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
    return freqdist.most_common(num)


freq_words('wordproc.py', 4, 10)
freq_words('wordproc.py', min=4, num=10)
freq_words('wordproc.py', num=10, min=4)


# 增加了 verbose 参数，可以设置为调试开关
def freq_words(file, min=1, num=10, verbose=False):
    freqdist = nltk.FreqDist()
    if verbose: print('Opening', file)
    with open(file) as f:
        text = f.read()
    if verbose: print('Read in %d characters' % len(file))
    for word in nltk.word_tokenize(text):
        if len(word) >= min:
            freqdist[word] += 1
            if verbose and freqdist.N() % 100 != 0: print('.')
    if verbose: print()
    return freqdist.most_common(num)


freq_words('wordproc.py', 4, 10, True)


# P169 4.6 程序开发
# Structure of a Python Module, Python 模块的结构
# Python 3 不支持 __file__
# 每个py文件都需要有注释头，包括：模块标题 和 作者信息。
# 模块级的 docstring，三重引号的多行字符串
# 模块需要的所有导入语句，然后是所有全局变量，接着是组成模块主要部分的一系列函数的定义

# P171 多模块程序
# 通过将工作分成几个模块和使用 import 语句访问其他模块定义的函数，可以保持各个模块的简单性
# 并且易于维护

# P172 错误代码，result 只会被初始化一次
def find_words(text, wordlength, result=[]):
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# 重复执行得到错误的结果，因为result默认不会每次调用都初始化为空列表
find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 3)
find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 2, ['ur'])
find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 3)
find_words(['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat'], 2)


# 使用None为占位符，None 是不可变对象，就可以正确初始化
# 不可变对象包含：整型、浮点型、字符串、元组
def find_words(text, wordlength, result=None):
    if not result: result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


def find_words(text, wordlength, result=()):
    if result == (): result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# 使用与默认值不同类型的对象作为默认值也可以
def find_words(text, wordlength, result=object):
    if result is object: result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# P173 调试技术：使用IDE编辑工具就不用命令行调试器了

# P174 防御性编程
# 测试驱动开发

# P175 4.7 算法设计
# P176 递归与迭代，两种解决方案各有利弊。递归更容易理解，迭代速度更快
def factorial1(n):
    result = 1
    for i in range(n):
        result *= (i + 1)
    return result


def factorial2(n):
    if n == 1:
        return 1
    else:
        return n * factorial2(n - 1)


def size1(s):
    return 1 + sum(size1(child) for child in s.hyponyms())


def size2(s):
    layer = [s]
    total = 0
    while layer:
        total += len(layer)
        layer = [h for c in layer for h in c.hyponyms()]
    return total


from nltk.corpus import wordnet as wn

dog = wn.synset('dog.n.01')
size1(dog)
size2(dog)


# Ex4-6 构建一个字母查找树：一个递归函数建立一个嵌套的字典结构，每一级嵌套包含给定前缀的所有单词
# 而子查找树含有所有可能的后续词
def insert(trie, key, value):
    if key:
        first, rest = key[0], key[1:]
        if first not in trie:
            trie[first] = {}
        insert(trie[first], rest, value)
    else:
        trie['value'] = value


trie = {}
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylist')
trie = dict(trie)
trie['c']['h']['a']['t']['value']
pprint.pprint(trie, width=40)


# P179 空间与时间的平衡，通过对文档索引集合，提高搜索速度，然后再对文档展开搜索，减少搜索准备
def raw(file):
    contents = open(file).read()
    contents = re.sub(r'<.*?>', ' ', contents)
    contents = re.sub('\s+', ' ', contents)
    return contents


def snippet(doc, term):
    text = ' ' * 30 + raw(doc) + ' ' * 30
    pos = text.index(term)
    return text[pos - 30:pos + 30]


print('Building Index...')
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w, f) for f in files for w in raw(f).split())

query = ''
while query != 'quit':
    query = input('query> ')
    if query in idx:
        for doc in idx[query]:
            print(snippet(doc, query))
    else:
        print('Not found')


# Ex4-8 预处理已经标的语料库数据，将所有的词和标都转换成整数
def preprocess(tagged_corpus):
    words = set()
    tags = set()
    for sent in tagged_corpus:
        for word, tag in sent:
            words.add(word)
            tags.add(tag)
    wm = dict((w, i) for (i, w) in enumerate(words))
    tm = dict((t, i) for (i, t) in enumerate(tags))
    return [[(wm[w], tm[t]) for (w, t) in sent] for sent in tagged_corpus]


# 使用timeit模块检测执行速度
# Timer类有两个参数：一个是多次执行的代码；一个是只在开始执行一次的设置代码。
# 例子：整数的链表 和 整数的集合 模拟10万个项目的词汇表
# 测试声明将产生随机项，它有50%的机会出现在词汇表中。
from timeit import Timer

vocab_size = 100000
setup_list = 'import random; vocab=range(%d)' % vocab_size
setup_set = 'import random; vocab=set(range(%d))' % vocab_size
statement = 'random.randint(0, %d) in vocab' % (vocab_size * 2)
# 以前的Python集合比列表快，现在几乎没有差别
print(Timer(statement, setup_list).timeit(1000))
print(Timer(statement, setup_set).timeit(1000))
vocab = range(vocab_size)


# P181 动态规划：是在自然语言处理中广泛使用的算法。
# 解决的问题内部包含了多个重叠的子问题。
# 算法可以避免重复计算这些子问题，而是简单地将它们的计算结果存储在一个查找表中。
# Ex4-9 4种方法计算梵文旋律：迭代、自底向上的动态规划、自上而下的动态规划、内置默记法
# 迭代算法
def virahanka1(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka1(n - 1)]
        l = ["L" + prosody for prosody in virahanka1(n - 2)]
        return s + l


# 自底向上的动态规划
def virahanka2(n):
    lookup = [[""], ["S"]]
    for i in range(n - 1):
        s = ["S" + prosody for prosody in lookup[i + 1]]
        l = ["L" + prosody for prosody in lookup[i]]
        lookup.append(s + l)
    return lookup[n]


# 自上而下的动态规划
def virahanka3(n, lookup={0: [""], 1: ["S"]}):
    if n not in lookup:
        s = ["S" + prosody for prosody in virahanka3(n - 1)]
        l = ["L" + prosody for prosody in virahanka3(n - 2)]
        lookup[n] = s + 1
    return lookup[n]


# 内置默记法
from nltk import memoize


@memoize
def virahanka4(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n - 1)]
        l = ["L" + prosody for prosody in virahanka4(n - 2)]
        return s + l


from functools import lru_cache


@lru_cache(1000)
def virahanka5(n):
    if n == 0:
        return [""]
    elif n == 1:
        return ["S"]
    else:
        s = ["S" + prosody for prosody in virahanka4(n - 1)]
        l = ["L" + prosody for prosody in virahanka4(n - 2)]
        return s + l


# 递归计算中会有重复计算的部分
virahanka1(4)
# 将较小的实例计算结果填充到表格中，一旦得到感兴趣的值就停止，
# 原则是解决较大问题之前先解决较小的问题，这就是自下而上的动态规划
# 某些计算得到的子问题在解决主问题时可能并不需要，从而造成浪费
virahanka2(4)
# 自上而下的可以避免计算不需要的子问题带来的浪费
virahanka3(4)
# 使用Python的装饰器模式引入的缓存机制
# 由functools.lru_cache实现的Python的memoization比我们的专用memoize函数更全面，就像你在CPython源代码中看到的一样
virahanka4(4)
virahanka5(4)

# P183 4.8 Python库的样例
# Matplotlib

from numpy import arange
from matplotlib import pyplot

colors = 'rgbcmyk'  # red, green, blue, cyan, magenta, yellow, black


def bar_chart(categories, words, counts):
    'Plot a bar chart showing counts for each word by category'
    ind = arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = pyplot.bar(ind + c * width, counts[categories[c]], width, color=colors[c % len(colors)])
        bar_groups.append(bars)
    pyplot.xticks(ind + width, words)
    pyplot.legend([b[0] for b in bar_groups], categories, loc='upper left')
    pyplot.ylabel('Frequency')
    pyplot.title('Frequency of Six Modal Verbs by Genre')
    pyplot.show()


genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
    (genre, word) for genre in genres
    for word in nltk.corpus.brown.words(categories=genre)
    if word in modals)

counts = {}
for genre in genres:
    counts[genre] = [cfdist[genre][word] for word in modals]

bar_chart(genres, modals, counts)

import sys

sys.setrecursionlimit(100000)  # 设置递归深度为100000，避免递归深度不够，导致pyplot.savefig()报错

from matplotlib import use, pyplot

use('Agg')
pyplot.savefig('modals.png')  # RecursionError: maximum recursion depth exceeded，因为递归深度超过1000
print('Content-Type: text/html')
print()
print('<html><body>')
print('<img src="modals.png"/>')
print('</body></html>')

# P182 NetworkX包定义和操作由节点和边组成的结构（称为图）。
# 使用NetworkX 和 Matplotlib 结合来可视化WordNet的网络结构（语义网络）

import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn


def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)


def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G


def graph_draw(graph):
    nx.draw(graph, node_size=[16 * graph.degree(n) for n in graph],
            node_color=[graph.depth[n] for n in graph], with_labels=False)
    matplotlib.pyplot.show()


dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)

# CSV
import csv

input_file = open('lexicon.csv', 'rb')  # 不能使用'b'，即bytes格式读取会报错
input_file = open('lexicon.csv')
print(list(csv.reader(input_file)))
input_file.seek(0)  # 文件内容读出后，需要重新将指针归零，再能再次读出数据
for row in csv.reader(input_file):
    print(row)

# NumPy 提供了多维数组对象
from numpy import array

cube = array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
              [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
              [[6, 6, 6], [7, 7, 7], [8, 8, 8]]])
cube[1, 1, 1]
cube[2].transpose()
cube[2, 1:]

# Numpy 提供了线性代数函数。
# 可以进行矩阵的奇异值分解，应用在潜在语义分析中，帮助识别文档集合中的隐含概念。
from numpy import linalg

a = array([[4, 0], [3, -5]])
u, s, vt = linalg.svd(a)
print(u)
print(s)
print(vt)
