# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0505.py
@Version    :   v0.1
@Time       :   2020-11-20 11:24
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk
from nltk.corpus import brown

# -   词性标注（parts-of-speech tagging，POS tagging）：简称标注。将词汇按照它们的词性（parts-of-speech，POS）进行分类并对它们进行标注
# -   词性：也称为词类或者词汇范畴。
# -   标记集：用于特定任务标记的集合。
from tools import show_subtitle

# Ch5 分类和标注词汇
# 1.  什么是词汇分类，在自然语言处理中它们如何使用？
# 2.  对于存储词汇和它们的分类来说什么是好的 Python 数据结构？
# 3.  如何自动标注文本中每个词汇的词类？

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
brown_tagged_words = brown.tagged_words(categories='news')
brown_words = brown.words(categories='news')

# 5.5 N元语法标注器
# xxxTagger() 只能使用 sent 作为训练语料

# 5.5.1 一元标注器，统计词料库中每个单词标注最多的词性作为一元语法模型的建立基础
# 使用训练数据来评估一元标注器的准确度
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print("unigram_tagger.tag(brown_sents[2007])= ", unigram_tagger.tag(brown_sents[2007]))
unigram_tagger.evaluate(brown_tagged_sents)

# 5.5.2 将数据分为 训练集 和 测试集
# 使用训练数据来训练一元标注器，使用测试数据来评估一元标注器的准确度
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

# 5.5.3 更加一般的N元标注器
# 二元标注器
bigram_tagger = nltk.BigramTagger(train_sents)

# 标注训练集中数据
show_subtitle("bigram_tagger.tag(train_sents[2007])")
print(bigram_tagger.tag(brown_sents[2007]))

# 标注测试集中数据
show_subtitle("bigram_tagger.tag(brown_sents[4203])")
print(bigram_tagger.tag(brown_sents[4203]))

bigram_tagger.evaluate(test_sents)  # 整体准确度很低，是因为数据稀疏问题

# 5.5.4 组合标注器，效果更差，为什么？
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t1.evaluate(test_sents)

t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)  # 这个效果最好

t3 = nltk.TrigramTagger(train_sents, backoff=t2)
t3.evaluate(test_sents)

t2 = nltk.BigramTagger(train_sents, cutoff=1, backoff=t1)
t2.evaluate(test_sents)

# cutoff=15时，准确率高，可见上下文并不能真正提示单词标注的内在规律
t3 = nltk.TrigramTagger(train_sents, cutoff=15, backoff=t2)
t3.evaluate(test_sents)

# 5.5.5 标注未知的单词
# 对于生词。可以使用回退到正则表达式标注器或者默认标注器，但是都无法利用上下文。

# 5.5.6 标注器的存储
from pickle import dump, load

text = """The board's action shows what free enterprise
    is up against in our complex maze of regulatory laws ."""
tokens = text.split()

output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()
print("t2.tag(tokens)= ", t2.tag(tokens))
print("t2.evaluate(test_sents)= ", t2.evaluate(test_sents))

input = open('t2.pkl', 'rb')
t2_bak = load(input)
input.close()

print("t2_bak.tag(tokens)= ", t2_bak.tag(tokens))
print("t2_bak.evaluate(test_sents)= ", t2_bak.evaluate(test_sents))

# 5.5.7 N元标注器的性能边界（上限）
# 一种方法是寻找有歧义的单词的数目，大约有1/20的单词可能有歧义
# cfd无法正确赋值，因为有些句子的长度少于3个单词，影响了trigrams()函数的正确运行
cfd = nltk.ConditionalFreqDist(
        ((x[1], y[1], z[0]), z[1])
        for sent in brown_tagged_sents
        if len(sent) >= 3
        for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [
        c
        for c in cfd.conditions()
        if len(cfd[c]) > 1
]
sum(
        cfd[c].N()
        for c in ambiguous_contexts
) / cfd.N()

# Colquitt 就是那个错误的句子，在ca01文本文件中可以找到
for i, sent in enumerate(brown_tagged_sents[:3]):
    show_subtitle(str(i))
    print("len(sent)= ", len(sent))
    print("tag(sent)= ", sent)
    if len(sent) >= 3:
        for x, y, z in nltk.trigrams(sent):
            print(x[0], y[0], z[0], x[1], y[1], z[1])

# 一种方法是研究被错误标记的单词
# ToDo: 可是显示出来的结果根本没有可视性呀？
test_tags = [
        tag
        for sent in brown.sents(categories='editorial')
        for (word, tag) in t2.tag(sent)
]
gold_tags = [
        tag
        for (word, tag) in brown.tagged_words(categories='editorial')
]
print(nltk.ConfusionMatrix(gold_tags, test_tags))

# 跨句子边界的标注
# 使用三元标注器时，跨句子边界的标注会使用上个句子的最后一个词+标点符号+这个句子的头一个词
# 但是，两个句子中的词并没有相关性，因此需要使用已经标注句子的链表来训练、运行和评估标注器
# Ex5-5 句子层面的N-gram标注
# 前面的组合标注器已经是跨句子边界的标注

# 5.6 基于转换的标注
# n-gram标注器存在的问题：
# 1）表的大小（语言模型），对于trigram表会产生巨大的稀疏矩阵
# 2）上下文。n-gram标注器从上下文中获得的唯一信息是标记，而忽略了词本身。
# 在本节中，利用Brill标注，这是一种归纳标注方法，性能好，使用的模型仅有n-gram标注器的很小一部分。
# Brill标注是基于转换的学习，即猜想每个词的标记，然后返回和修正错误的标记，陆续完成整个文档的修正。
# 与n-gram标注一样，需要监督整个过程，但是不计数观察结果，只编制一个转换修正规则链表。
# Brill标注依赖的原则：规则是语言学可解释的。因此Brill标注可以从数据中学习规则，并且也只记录规则。
# 而n-gram只是隐式的记住了规律，并没有将规律抽象出规则，从而记录了巨大的数据表。
# Brill转换规则的模板：在上下文中，替换T1为T2.
# 每一条规则都根据其净收益打分 = 修正不正确标记的数目 - 错误修改正确标记的数目
from nltk.tbl import demo as brill_demo

brill_demo.demo()
print(open('errors.out').read())

# 5.7 如何确定一个词的分类（词类标注）
# 语言学家使用形态学、句法、语义来确定一个词的类别

# 5.7.1 形态学线索：词的内部结构有助于词类标注。

# 5.7.2 句法线索：词可能出现的典型的上下文语境。

# 5.7.3 语义线索：词的意思

# 5.7.4 新词（未知词）的标注：开放类和封闭类

# 5.7.5 词性标记集中的形态学
# 普通标记集捕捉的构词信息：词借助于句法角色获得的形态标记信息。
# 大多数词性标注集都使用相同的基本类别。更精细的标记集中包含更多有关这些形式的信息。
# 没有一个“正确的方式”来分配标记，只能根据目标不同而产生的或多或少有用的方法

# 5.8 小结
# -   词可以组成类，这些类称为词汇范畴或者词性。
# -   词性可以被分配短标签或者标记
# -   词性标注、POS标注或者标注：给文本中的词自动分配词性的过程
# -   语言词料库已经完成了词性标注
# -   标注器可以使用已经标注过的语料库进行训练和评估
# -   组合标注方法：把多种标注方法（默认标注器、正则表达式标注器、Unigram标注器、N-gram标注器）利用回退技术结合在一起使用
# -   回退是一个组合模型的方法：当一个较为专业的模型不能为给定内容分配标记时，可以回退到一个较为一般的模型
# -   词性标注是序列分类任务，通过利用局部上下文语境中的词和标记对序列中任意一点的分类决策
# -   字典用来映射任意类型之间的信息
# -   N-gram标注器可以定义为不同数值的n，当n过大时会面临数据稀疏问题，即使使用大量的训练数据，也只能够看到上下文中的一部分
# -   基于转换的标注包括学习一系列的“改变标记s为标记t在上下文c中”形式的修复规则，每个规则都可以修复错误，但是也可能会引入新的错误
