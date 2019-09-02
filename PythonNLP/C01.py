import nltk

from nltk.book import *
from nltk.corpus import udhr

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1. 语言计算：文本和词汇
# 1.3. 搜索文本

# 查找 monstrous 词出现的位置
text1.concordance('monstrous')
text1.concordance('contemptible')

# 与 monstrous 使用一个词汇相似的上下文的单词
# 'most monstrous and'
# 'most contemptible and'
text1.similar('monstrous')


text1.concordance('christian')
text1.concordance('contemptible')

text1.common_contexts(['monstrous', 'the'])
# text2中'a very pretty'和'a monstrous pretty'，下句输出'a_pretty'
# 即'monstrous'和'very'共用两个词汇的上下文
text2.common_contexts(['monstrous', 'very'])
text2.concordance('a very pretty')
text2.concordance('monstrous')

# 文章中下列词汇在文章中使用的分布图
text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America'])

# NLTK 3.0的版本已经放弃了这个功能
text3.generate()

sorted(set(text3))


# 每个单词的平均使用次数
def lexical_diversity(text):
    return len(text) / len(set(text))


lexical_diversity(text3)


# 特定单词在文本中占据的百分比
def percentage(word, text):
    return 100 * text.count(word) / len(text)


# 'smote'单词出现在text3中的次数，已经在占所有单词数的百分比`
percentage('smote', text3)

# 2. Python 将文本当作词链表
sent1 = ['Call', 'me', 'Ishmael', '.']
len(sent1)
lexical_diversity(sent1)
ex1 = ['Monty', 'Python', 'and', 'the', 'Holy', 'Grail']
sorted(ex1)
len(set(ex1))
ex1.count('the')
ex2 = ['!']
ex3 = ex1 + ex2
sent1.append('Some')

# 索引列表
text4[173]
text4.index('awaken')

# 索引切片
text5[16715:16735]
text6[1600:1625]

# 索引从零开始
sent = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10']
sent[0]
sent[9]
sent[10]
sent[5:8]
sent[:3]
sent[7:]
sent[-1:]
sent[:-1]
sent[0] = 'First'
sent[9] = 'Last'
len(sent)
sent[1:9]

# 字符串
string = 'MontyPython'
string[0]
string[:4]
string[0] = 'm'  # TypeError: 'str' object does not support item assignment
string * 2
string + '!'
' '.join(['Monty', 'Python'])
'Monty Python'.split()

# 3. 通过简单的统计来计算语言
saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']
# set()函数会把相同的单词合并，集合没有重复项
sorted(saying)
sorted(set(saying))

tokens = set(saying)
tokens = sorted(tokens)
tokens[-3:]

# 3.1. 频率分布
fdist1 = FreqDist(text1)
fdist1
fdist1.most_common(50)
fdist1['whale']
fdist1.plot(50, cumulative=True)
fdist1.hapaxes()  # 只出现一次的单词，低频词
len(fdist1.hapaxes())
vocabulary1 = fdist1.keys()
vocabulary1[:50]  # 这个用法已经被废弃了

# 3.2. 细粒度的选择单词
# long_words = { w | w∈V & P(w) } = [w for w in V if p(w)]
V = set(text1)
words1 = [w for w in V if len(w) > 15]
sorted(words1)
len(words1)

fdist5 = FreqDist(text5)
words5 = [w for w in set(text5) if len(w) > 7 and fdist5[w] > 7]
sorted(words5)
len(words5)

# 3.3. 词语搭配 和 双连词
# 搭配：不经常在一起出现的词序列。例如：red wine 是一个搭配；the wine 不是一个搭配。
# 要获取搭配，需要从提取文本词汇中的词对（即双连词）开始。使用函数 bigrams() 实现。
doubleWords = list(bigrams(['more', 'is', 'said', 'than', 'done']))

text4.collocations()  # 搜索出文本中的双连词，（window_size的作用？）
# 3.4. 计算其他东西
[len(w) for w in text1]
fdist_word_len_1 = FreqDist([len(w) for w in text1])
fdist_word_len_1.plot()
fdist_word_len_1.tabulate()

fdist_word_len_1.keys()
fdist_word_len_1.hapaxes()

fdist_word_len_1.N()
fdist_word_len_1.most_common()
fdist_word_len_1.max()
fdist_word_len_1[3]  # 给定样本的数目
fdist_word_len_1.freq(3)  # 给定样本在数据中的占比

for element in fdist_word_len_1.elements():
    print(element)

# 4. Python 的决策 与 控制
# 4.1. 条件
# P24 表1-3，数值比较运算符
# [w for w in text if condition]
[w for w in sent7 if len(w) < 4]
[w for w in sent7 if len(w) <= 4]
[w for w in sent7 if len(w) == 4]
[w for w in sent7 if len(w) != 4]

# P25 表1-4，词汇比较运算符
sorted([w for w in set(text1) if w.endswith('ableness')])  # 单词以'ableness'结尾
sorted([term for term in set(text4) if 'gnt' in term])  # 单词中包含'gnt'
sorted([item for item in set(text6) if item.istitle()])  # 首字母大写
sorted([item for item in set(sent7) if item.isdigit()])  # 单词是数字

sorted([w for w in set(text7) if '-' in w and 'index' in w])
sorted([wd for wd in set(text3) if wd.istitle() and len(wd) > 10])
sorted([w for w in set(sent7) if not w.islower()])
sorted([t for t in set(text2) if 'cie' in t or 'cei' in t])

# 4.2. 操作每个元素
[len(w) for w in text1]
[w.upper() for w in text1]

len(text1)
len(set(text1))
len([word.lower() for word in set(text1)])
len(set([word.lower() for word in text1]))
len(set([word.lower() for word in text1 if word.isalpha()]))

# 4.3. 嵌套代码块
word = 'cat'

if len(word) < 5:
    print('word length is less than 5')

if len(word) < 5:
    print('word length is greater than or equal to 5')

for word in ['Call', 'me', 'Ishmael', '.']:
    print(word)

# 条件循环
sent1 = ['Call', 'me', 'Ishmael', '.']
for xyzzy in sent1:
    if xyzzy.endswith('l'):
        print(xyzzy)

tricky=sorted(w for w in set(text2) if 'cie' in w or 'cei' in w)
for word in tricky:
    print(word, end=' ')

# 5. 自然语言理解的自动化处理
# 5.1. 词意消歧：相同的单词在不同的上下文中指定不同的意思
# 5.2. 指代消解：检测动词的主语和宾语
# 指代消解：确定代词或名字短语指的是什么？
# 语义角色标注：确定名词短语如何与动词想关联（如代理、受事、工具等）
# 5.3. 自动生成语言：如果能够自动地解决语言理解问题，就能够继续进行自动生成语言的任务，例如：自动问答和机器翻译。
# 5.4. 机器翻译：
# 文本对齐：自动配对组成句子
# 5.5. 人机对话系统：图1-5，简单的话音对话系统的流程架构
# 5.6. 文本的含义：文本含义识别（Recognizing Textual Entailment, RTE）
# 5.7. NLP的局限性
