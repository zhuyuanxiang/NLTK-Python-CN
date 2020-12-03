# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C08.py
@Version    :   v0.1
@Time       :   2020-11-30 16:39
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
import nltk

from tools import show_subtitle

# Ch8 分析句子结构
# 前面的章节重点关注词的处理：识别、分析结构、分配词汇类别、获取词汇的含义、识别词序列或者n-grams的模式
# 本章的学习目标：
# 1）使用形式化语法来描述无限的句子集合的结构
# 2）使用句法树来表示句子结构
# 3）使用解析器来分析句子，并且自动构建语法树

# 8.1 一些语法困境
# 8.1.1 语言数据和无限的可能性
# 本章中将采用“生成文法”的形式化框架，其中一种“语言”被认为仅仅是所有合乎文法的句子的大集合，
# 而文法只是一个形式化符号，可以用于“生成”这个集合的成员。文法使用 “S → S and S” 形式的递归产生式

# 8.1.2 普遍存在的歧义
groucho_grammar = nltk.CFG.fromstring('''
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | "I"
VP -> V NP | VP PP
Det -> "an" | "my"
N -> "elephant" | "pajamas"
V -> "shot"
P -> "in"
''')

# 基于一种文法解析句子，可能会解析出两种结构
sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(groucho_grammar)  # 图解析
for i, tree in enumerate(parser.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

# 8.2 文法的用途
# 8.2.1 超越 n-grams
# 成分结构是词与词结合在一起组成的单元。
# 在符合语法规则的句子中的词序列是可以被一个更小的词序列替代，并且这个词序列不会导致句子不合语法规则
# 形成单元的每个序列都可以被单独的词替换。
# 句子长度是任意的，因此短语结构树的深度也是任意的。因为Sec7.4只能产生有限深度的结构，所以分块方法并不适合用于句法分析。

# 8.3 上下文无关文法（context-free grammars，CFG）
# 8.3.1 一种简单的文法

# Ex8-1 一个简单的上下文无关文法的例子
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
Det -> "a" | "an" | "the" | "my" | "The"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
""")

# 句子剖析会出现两个符合文法规则的结果，称为结构上有歧义。这个歧义称为介词短语附着歧义。
sent = 'The dog saw a man in the park'.split()
rd_parser = nltk.RecursiveDescentParser(grammar1)  # 递归下降解析器 RecursiveDescentParser()
for i, tree in enumerate(rd_parser.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

nltk.app.rdparser()  # 通过这个演示可以辅助理解从顶向下的回溯策略的句法剖析过程

# 8.3.2 编写自己的文法
# 在文本文件创建和编辑语法会更加文法，然后可以利用函数加载到NLTK中进行解析
# grammar1 = nltk.data.load('file:D:/mygrammar1.cfg')
grammar1 = nltk.data.load('mygrammar.cfg')
sent = "Mary saw Bob".split()
sent = "Mary saw Bob's mother".split()  # 无法解析的句子
rd_parser = nltk.RecursiveDescentParser(grammar1)  # trace = 2 不知道有何作用
for i, tree in enumerate(rd_parser.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

for p in grammar1.productions():
    print(p)

# 8.3.3 句法结构中的递归
# RecursiveDescentParser()无法处理形如X→XY的左递归产生式

# Ex-2：递归的上下文无关文法
grammar2 = nltk.CFG.fromstring("""
S  -> NP VP
NP -> Det Nom | PropN
Nom -> Adj Nom | N
VP -> V Adj | V NP | V S | V NP PP
PP -> P NP
PropN -> 'Buster' | 'Chatterer' | 'Joe'
Det -> 'the' | 'a'
N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
P -> 'on'
""")

sent = 'the angry bear chased the frightened little squirrel'.split()
sent = 'Chatterer said Buster thought the tree was tall'.split()
rd_parser = nltk.RecursiveDescentParser(grammar2)
for i, tree in enumerate(rd_parser.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

# 8.4 上下文无关文法分析
# 解析器根据文法产生式处理输入的句子，并且建立一个或者多个符合文法的组成结构
# 文法是一个格式良好的声明规则——实际上只是一个字符串，而不是程序。
# 解析器是文法的解释程序，用于搜索所有的符合文法的树的空间，并且找出一棵与句子匹配的语法树
# 两种分析算法：
# 1) 自顶向下的递归下降分析，主要缺点：
#       左递归产生式，如：NP→NP PP，会进入死循环
#       解析器在处理不符合输入句子的词和结构时会浪费许多时间
#       回溯过程中可能会丢弃分析过的成分，需要再次重建
# 2）自底向上的移进归约分析，只建立与输入中的词对应的结构，对于每个子结构只建立一次
#       反复将下一个输入词捡到堆栈，叫做移位操作
#       替换前n项为一项的操作，叫做归约操作
# 移进—归约解析器 ShiftReduceParser()
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
Det -> "a" | "an" | "the" | "my" | "The"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
""")

sent = 'Mary saw a dog'.split()
sr_parser = nltk.ShiftReduceParser(grammar1)
# sr_parser = nltk.ShiftReduceParser(grammar1,trace = 2)
for i, tree in enumerate(rd_parser.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

nltk.app.rdparser()  # 通过这个演示可以辅助理解自顶向下的回溯策略的句法剖析过程
nltk.app.srparser()  # 通过这个演示可以辅助理解自底向上的称进归约的句法剖析过程

# 8.4.3 左角落解析器：是 自顶向下 和 自底向上 方法的混合体，是一个带有自底向上过滤的自顶向下的解析器
# 左角落解析器，不会陷入左递归产生式死循环。

# 8.4.4 符合语句规则的子串表（WFST）
# 上述简单的解析器都存在完整性和效率问题，下面将基于图表分析：即利用动态规划算法来解决这些问题
# 动态规划算法存储中间结果，并且在适当的时候重用，从而显著提高了效率。
# WFST的缺点：
# * WFST本身不是一个分析树
# * 每个非词汇文法生产式都必须是二元的
# * 作为一个自下而上的文法，潜在地存在着浪费，因为它会在不符合文法的地方提出成分，后面又会放弃掉错误的成分
# * WFST并不能表示句子中的结构歧义（如两个动词短语的读取）

# 可以在文法中直接查找文本中单词所属类别
# lhs : left-hand-side; rhs : right-hand-side
text = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
productions = groucho_grammar.productions(rhs=text[3])
productions[0].lhs()


# Ex8-3 使用符合语句规则的子串表接收器
def init_wfst(tokens, grammar):
    numtokens = len(tokens)
    # 生成 None 组成的 wfst 列表数组
    wfst = [[None for i in range(numtokens + 1)] for j in range(numtokens + 1)]
    for i in range(numtokens):
        productions = grammar.productions(rhs=tokens[i])
        wfst[i][i + 1] = productions[0].lhs()
    return wfst


def complete_wfst(wfst, tokens, grammar, trace=False):
    index = dict(
        (production.rhs(), production.lhs())
        for production in grammar.productions())
    num_tokens = len(tokens)
    for span in range(2, num_tokens + 1):
        for start in range(num_tokens + 1 - span):
            end = start + span
            for mid in range(start + 1, end):
                nt1, nt2 = wfst[start][mid], wfst[mid][end]
                if nt1 and nt2 and (nt1, nt2) in index:
                    wfst[start][end] = index[(nt1, nt2)]
                    if trace:
                        print("[%s] %3s [%s] %3s [%s] ==> [%s] %3s [%s]" %
                              (start, nt1, mid, nt2, end, start, index[(nt1, nt2)], end))
    return wfst


def display(wfst, tokens):
    print('\nWFST ' + ' '.join(
        [
            ("%-4d" % idx)
            for idx in range(1, len(wfst))]))
    for idx in range(len(wfst) - 1):
        print("%d   " % idx, end=" ")
        for j in range(1, len(wfst)):
            print("%-4s" % (wfst[idx][j] or '.'), end=" ")
        print()


tokens = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
wfst0 = init_wfst(tokens, groucho_grammar)
display(wfst0, tokens)

wfst1 = complete_wfst(wfst0, tokens, groucho_grammar)
display(wfst1, tokens)

# 显示剖析过程
wfst1 = complete_wfst(wfst0, tokens, groucho_grammar, trace=True)
display(wfst1, tokens)

nltk.app.chartparser()

# 8.5 依存关系 和 依存文法
# 短语结构文法：描述句子中的词和词序列的结合方式
# 依存文法：描述词与其他词之间的关系
# 依存关系是一个中心词与其从属之间二元非对称关系。
# 一个句子的中心词通常是动词，其他词直接依赖于中心词或者通过某些路径依赖于中心词

# 下面是NLTK为依存文法编码的一种方式，只能捕捉依存关系信息，不能指定依存关系的类型
groucho_dep_grammar = nltk.DependencyGrammar.fromstring("""
'shot' -> 'I' | 'elephant' | 'in'
'elephant' -> 'an' | 'in'
'in' -> 'pajamas'
'pajamas' -> 'my'
""")
print(groucho_dep_grammar)

# 依存关系图是一个投影，若所有的词都按照线性顺序书写，则用边连接这些词并且保证所有的边不交叉。
# 一个词及其所有子节点在句子中形成了一个连续的词序列。
pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
sent = 'I shot an elephant in my pajamas'.split()
tree = []
for i, tree in enumerate(pdp.parse(sent)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)
tree.draw()

# 非投影的依存关系
# 在一个成分C中决定哪个是中心词H，哪个是依赖D：
# 1）H决定类型C的分布；或者说C的外部句法属性取决于H
# 2）H定义C的语义类型
# 3）H必须存在，而D是可选的
# 4）H选择D并且决定它是必须的还是可选的
# 5）D的形态由H决定

# 8.5.1 配价与词汇
# 动词的配价（Valency）指动词在句子中对名词或者名词性成分的支配能力。
# 动词的价就是句中的核心动词可以直接支配的名词或者名词性成分的数量。
# 表8-3中的动词被认为具有不同的配价。配价限制不仅适用于动词，也适应于其他类的中心词。

# 8.5.2 扩大规模
# “玩具文法”：用于演示分析过程中关键环节的小型文法。将文法扩大到覆盖自然语言中的大型语料库非常困难。
# 比较成功的文法项目
# * 词汇功能语法（LFG） Pargram项目
# * 中心词驱动短语结构文法（HPSG）LinGO项目
# * 邻接着文法XTAG的词汇化树项目

# 8.6 文法开发
# 解析器根据短语结构文法在句子上建立树。

# 8.6.1 树库 和 文法：使用宾州树库

from nltk.corpus import treebank

t = treebank.parsed_sents('wsj_0001.mrg')[0]
print(t)


# Ex 8-4 搜索树库找出句子的补语
def filter(tree):
    child_nodes = [child.label() for child in tree if isinstance(child, nltk.Tree)]
    return (tree.label() == 'VP') and ('S' in child_nodes)


VPS = [
    subtree
    for tree in treebank.parsed_sents()
    for subtree in tree.subtrees(filter)]
print(VPS[0])
VPS[1].draw()

# Prepositional Phrase Attachment Corpus. 介词短语附着语料库，是特别动词配价的信息源
# 搜索语料库，找出具有固定介词和名词和介词短语对，其中介绍短语附着到VP还是NP由选择的动词决定
from collections import defaultdict

entries = nltk.corpus.ppattach.attachments('training')
table = defaultdict(lambda: defaultdict(set))
for entry in entries:
    key = entry.noun1 + '-' + entry.prep + '-' + entry.noun2
    table[key][entry.attachment].add(entry.verb)

for key in sorted(table):
    if len(table[key]) > 1:
        print(key, 'N: ', sorted(table[key]['N']), 'V:', sorted(table[key]['V']))

print("key=", key)
print("table[key]= ", table[key])
print("len(table[key])= ", len(table[key]))
print("table['zip-in-way']= ", table['zip-in-way'])
print("table['access-to-AZT']= ", table['access-to-AZT'])
print("table['offer-from-group']= ", table['offer-from-group'])

# 现代汉语中央研究院平衡语料库中的10000句已经分析的句子
nltk.corpus.sinica_treebank.parsed_sents()[3450].draw()

# 8.6.2 有害的歧义

grammar = nltk.CFG.fromstring("""
S -> NP V NP
NP -> NP Sbar
Sbar -> NP V
NP -> 'fish'
V -> 'fish'
""")

# 当fish的数量为（3，5，7...），分析树的数量是（1，2，5...），这是Catalan数
tokens = ['fish'] * 5
cp = nltk.ChartParser(grammar)
for i, tree in enumerate(cp.parse(tokens)):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)


# 8.6.3 加权语法

# Ex8-5: 宾州树库样本中 give 和 gave 的用法
# 检查所有包含 give 介词格和双宾语结构的实例
def give(t):
    result = t.label() == 'VP' and len(t) > 2 and t[1].label() == 'NP'  # give 双宾语结构
    result = result and (t[2].label() == 'PP-DTV' or t[2].label() == 'NP')  # give 介绍格
    result = result and ('give' in t[0].leaves() or 'gave' in t[0].leaves())  # give和gave的用法
    return result


def sent(tree):
    return ' '.join(token for token in tree.leaves() if token[0] not in '*-0')


def print_node(t, width):
    output = '%s %s: %s / %s: %s' % (
        sent(t[0]), t[1].label(), sent(t[1]), t[2].label(), sent(t[2]))
    if len(output) > width:
        output = output[:width] + '...'
    print(output)


for tree in nltk.corpus.treebank.parsed_sents():
    for t in tree.subtrees(give):
        print_node(t, 72)
print(t)

# Ex8-6: 定义一个概率上下文无关文法（PCFG）
# 只是演示了一个概率上下文无关文法的作用，也是个玩具文法
grammar = nltk.PCFG.fromstring("""
S    -> NP VP              [1.0]
VP   -> TV NP              [0.4]
VP   -> IV                 [0.3]
VP   -> DatV NP NP         [0.3]
TV   -> 'saw'              [1.0]
IV   -> 'ate'              [1.0]
DatV -> 'gave'             [1.0]
NP   -> 'telescopes'       [0.8]
NP   -> 'Jack'             [0.2]
""")

viterbi_parser = nltk.ViterbiParser(grammar)
for i, tree in enumerate(viterbi_parser.parse(['Jack', 'saw', 'telescopes'])):
    show_subtitle(f"第 {i + 1} 个结构")
    print(tree)

# 8.7 小结
# * 句子的内部结构用树来表示。组织结构的显著特点是：递归、中心词、补语和修饰语
# * 文法是可能句子的集合的紧凑型特性：一棵树是符合语法规则的，或者文法是可以授权给一棵树的
# * 文法是一种用于描述给定短语是否可以被分配给特定成分或者依存结构的形式化模型
# * 给定一组句法类别，上下文无关文法使用生产式表示某个类型A的短语是如何被分析成较小的序列α1…αn的
# * 依存文法使用产生式指定给定的中心词的依赖是什么
# * 当句子有一个以上的文法分析时，就会产生句法歧义（如介词短语附着歧义）
# * 解析器是寻找一个或者多个与符合语法规则句子相对应树的程序
# * 下降递归解析器是一个简单的自顶而下的解析器，利用文法产生式，递归可扩展的开始符号，并且深度匹配输入的句子。
#       下降递归解析器不能处理左递归产生式。
#       下降递归解析器只能盲目扩充类别而不检查是否与输入字符串兼容，导致效率低下。
# * 称进——规约解析器是一个简单的自底向上的解析器，输入被移到堆栈中，并且尝试匹配堆栈顶部的项目和文法产生式右边的部分。
#       移进——规约解析器，哪怕句子有效的解析确定存在，也不能保证为输入有效的解析。
#       移进——规约解析器，建立子结构，但是不检查它们是否与全部文法一致。
