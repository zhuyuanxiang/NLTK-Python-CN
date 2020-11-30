# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0701.py
@Version    :   v0.1
@Time       :   2020-11-30 12:42
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# Ch07 从文本提取信息
# 学习目标
# 1) 从非结构化文本中提取结构化数据
# 2） 识别一个文本中描述的实体和关系
# 3） 使用语料库来训练和评估模型

# 7.1 信息提取
# 从文本中获取意义的方法被称为「信息提取」
# 1） 从结构化数据中提取信息
# 2） 从非结构化文本中提取信息
#   * 建立一个非常一般的含义（Ch10）
#   * 查找文本中具体的各种信息
#       * 将非结构化数据转换成结构化数据
#       * 使用查询工具从文本中提取信息


# 7.1.1 信息提取结构：
# ｛原始文本（一串）｝→ 断句
# →｛句子（字符串列表）｝→ 分词
# →｛句子分词｝→ 词性标注
# →｛句子词性标注｝→ 实体识别
# →｛句子分块｝→ 关系识别
# →｛关系列表｝

# P283 图7-1，Ex7-1：信息提取结构元组（entity, relation, entity)
locs = [('Omnicom', 'IN', 'New York'),
        ('DDB Needham', 'IN', 'New York'),
        ('Kaplan Thaler Group', 'IN', 'New York'),
        ('BBDO South', 'IN', 'Atlanta'),
        ('Georgia-Pacific', 'IN', 'Atlanta')]
query = [
    e1
    for (e1, rel, e2) in locs
    if e2 == 'Atlanta'
]
print(query)


# 将 NLTK中的 句子分割器、分词器、词性标注器 连接在一起
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)  # 句子分割
    sentences = [
        nltk.word_tokenize(sent)
        for sent in sentences]  # 单词分割
    sentences = [
        nltk.pos_tag(sent)
        for sent in sentences]  # 词性标注
    return sentences


# 7.2 分块：用于实体识别的基本技术（P284 图7-2）
# * 小框显示词级标识符和词性标注
# * 大框表示组块（chunk），是较高级别的程序分块
# 分块构成的源文本中的片段不能重叠
# 正则表达式和N-gram方法分块；
# 使用CoNLL-2000分块语料库开发和评估分块器；

# 7.2.1 名词短语分块（NP-chunking，即“NP-分块”）寻找单独名词短语对应的块
# NP-分块是比完整的名词短语更小的片段，不包含其他的NP-分块，修饰一个任何介词短语或者从句将不包括在相应的NP-分块内。
# NP-分块信息最有用的来源之王是词性标记。
# Pattern Matches...
# <T> a word with tag T (where T may be a regexp).
# x? an optional x
# x+ a sequence of 1 or more x's
# x* a sequence of 0 or more x's
# x|y x or y
# . matches any character
# (x) Treats x as a group
# #x... Treats x... (to
# the end of the line) as a comment
# \C matches character C (useful when C is a special character like + or #)

# P285 Ex7-1 基于正则表达式的NP 分块器
# 使用分析器对句子进行分块
sentence = [('the', 'DT'),
            ('little', 'JJ'),
            ('yellow', 'JJ'),
            ('dog', 'NN'),
            ('barked', 'VBD'),
            ('at', 'IN'),
            ('the', 'DT'),
            ('cat', 'NN')]
# 定义分块语法
grammar = 'NP: {<DT>?<JJ>*<NN>}'
# 创建组块分析器
cp = nltk.RegexpParser(grammar)
# 对句子进行分块
result = cp.parse(sentence)
# 输出分块的树状图
print(result)
result.draw()

# 7.2.2. 标记模式
# 华尔街日报
sentence = [('another', 'DT'),
            ('sharp', 'JJ'),
            ('dive', 'NN'),
            ('trade', 'NN'),
            ('figures', 'NNS'),
            ('any', 'DT'),
            ('new', 'JJ'),
            ('policy', 'NN'),
            ('measures', 'NNS'),
            ('earlier', 'JJR'),
            ('stages', 'NNS'),
            ('Panamanian', 'JJ'),
            ('dictator', 'NN'),
            ('Manuel', 'NNP'),
            ('Noriega', 'NNP')]
grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)

# Grammar中输入语法，语法格式{<DT>?<JJ.*>*<NN.*>+}，不能在前面加NP:，具体可以参考右边的Regexps说明
# Development Set就是开发测试集，用于调试语法规则。绿色表示正确匹配，红色表示没有正确匹配。黄金标准标注为下划线
nltk.app.chunkparser()

# 7.2.3 用正则表达式分块（组块分析）
# Ex7-2 简单的名词短语分类器
sentence = [("Rapunzel", "NNP"),
            ("let", "VBD"),
            ("down", "RP"),
            ("her", "PP$"),
            ("long", "JJ"),
            ("golden", "JJ"),
            ("hair", "NN")]
# 两个规则组成的组块分析语法，注意规则执行会有先后顺序，两个规则如果有重叠部分，以先执行的为准
grammar = r'''
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
'''
print(nltk.RegexpParser(grammar).parse(sentence))
grammar = r'NP: {<[CDJ].*>+}'
print(nltk.RegexpParser(grammar).parse(sentence))
grammar = r'NP: {<[CDJNP].*>+}'
print(nltk.RegexpParser(grammar).parse(sentence))
grammar = r'NP: {<[CDJN].*>+}'
print(nltk.RegexpParser(grammar).parse(sentence))

# 如果模式匹配位置重叠，最左边的优先匹配。
# 例如：如果将匹配两个连贯名字的文本的规则应用到包含3个连贯名词的文本中，则只有前两个名词被分块
nouns = [('money', 'NN'), ('market', 'NN'), ('fund', 'NN')]
grammar = 'NP: {<NN><NN>}'
print("错误分块的结果= ", nltk.RegexpParser(grammar).parse(nouns))
grammar = 'NP: {<NN>+}'
print("正确分块的结果= ", nltk.RegexpParser(grammar).parse(nouns))

# 7.2.4 探索文本语料库：从已经标注的语料库中提取匹配特定词性标记序列的短语
grammar = 'CHUNK: {<V.*><TO><V.*>}'
cp = nltk.RegexpParser(grammar)
brown = nltk.corpus.brown
count = 0
for sent in brown.tagged_sents():
    if count < 10:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK':
                count += 1
                print(subtree)


# 定义一个搜索函数（一次性返回定义好的数据量）
def find_chunks(pattern):
    cp = nltk.RegexpParser(pattern)
    brown = nltk.corpus.brown
    count = 0
    for sent in brown.tagged_sents():
        if count < 10:
            tree = cp.parse(sent)
            for subtree in tree.subtrees():
                if subtree.label() == 'CHUNK':
                    count += 1
                    print(subtree)


grammar = 'CHUNK: {<V.*><TO><V.*>}'
find_chunks(grammar)

grammar = 'NOUNS: {<N.*>{4,}}'
find_chunks(grammar)


# 定义一个搜索函数（使用生成器）
def find_chunks(pattern):
    cp = nltk.RegexpParser(pattern)
    brown = nltk.corpus.brown
    for sent in brown.tagged_sents():
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK' or subtree.label() == 'NOUNS':
                yield subtree


grammar = 'CHUNK: {<V.*><TO><V.*>}'
for i, subtree in enumerate(find_chunks(grammar)):
    if i < 10:
        print(subtree)

grammar = 'NOUNS: {<N.*>{4,}}'
for i, subtree in enumerate(find_chunks(grammar)):
    if i < 10:
        print(subtree)

# 7.2.5. 添加缝隙：寻找需要排除的成分
sentence = [("the", "DT"),
            ("little", "JJ"),
            ("yellow", "JJ"),
            ("dog", "NN"),
            ("barked", "VBD"),
            ("at", "IN"),
            ("the", "DT"),
            ("cat", "NN")]

# 先分块，再加缝隙，才能得出正确的结果
grammar = r'''
    NP: 
        {<.*>+}         # Chunk everything
        }<VBD|IN>+{     # Chink sequences of VBD and IN
'''
print(nltk.RegexpParser(grammar).parse(sentence))

# 先加缝隙，再分块
# 结果是错的，只会得到一个块，效果与没有使用缝隙是一样的
grammar = r'''
    NP: 
        }<VBD|IN>+{     # Chink sequences of VBD and IN
        {<.*>+}         # Chunk everything
'''
print(nltk.RegexpParser(grammar).parse(sentence))

# 只分块，结果是错的
grammar = r'''
    NP: 
        {<.*>+}         # Chunk everything
'''
print(nltk.RegexpParser(grammar).parse(sentence))

# 7.2.6 分块的表示：标记与树状图
# 作为标注放分析之间的中间状态（Ref：Ch8），块结构可以使用标记或者树状图来表示
# 使用最为广泛的表示是IOB标记：I（Inside，内部）；O（Outside，外部）；B（Begin，开始）。
