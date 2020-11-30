# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0705.py
@Version    :   v0.1
@Time       :   2020-11-30 12:46
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk
import re

# Ch07 从文本提取信息
# 学习目标
# 1) 从非结构化文本中提取结构化数据
# 2） 识别一个文本中描述的实体和关系
# 3） 使用语料库来训练和评估模型

# 7.5 命名实体识别：识别所有文本中提及的命名实体。
# -   命名实体（Named Entity，NE）：是确切的名词短语，指特定类型的个体。
# -   命名实体识别（Named Entity Recognition，NER）即识别所有文本中提及的命名实体。
#     -   主要方法：查词典
#     -   主要困难：名称有歧义
#     -   主要手段：基于分类器进行分类
#     -   两个子任务
#         1.  确定NE的边界
#         2.  确定NE的类型

sent = nltk.corpus.treebank.tagged_sents()[22]
# NLTK提供的是已经训练好的可以识别命名实体的分类器
print(nltk.ne_chunk(sent))
# 使用nltk.ne_chunk()函数调用分类器，binary=True表示标注为NE，否则会添加类型标签，例如：PERSON，GPE等等。
print(nltk.ne_chunk(sent, binary=True))

# 7.6. 关系抽取：寻找指定类型的命名实体之间的关系
# 1）寻找所有（X，α，Y）形式的三元组，其中X和Y是指定类型的命名实体，α表示X和Y之间的关系的字符串
# 搜索包含词in的字符串
IN = re.compile(r'.*\bin')
# “(?!\b.+ing)”是一个否定预测先行断言，忽略如“success in supervising the transition of” 这样的字符串
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
        print(nltk.sem.rtuple(rel))

# nltk.sem NLTK （Semantic Interpretation Package）语义解释包
# 用于表达一阶逻辑的语义结构和评估集合论模型的公式
# This package contains classes for representing semantic structure in
# formulas of first-order logic and for evaluating such formulas in
# set-theoretic models.
from nltk.corpus import conll2002

vnv = '''
(
is/V|
was/V|
werd/V|
wordt/V
)
.*
van/Prep
'''
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for rel in nltk.sem.extract_rels('PER', 'ORG', doc, corpus='conll2002', pattern=VAN):
        # 抽取具备特定关系的命名实体
        clause = nltk.sem.clause(rel, relsym='VAN')
        # print(nltk.sem.clause(rel, relsym='VAN'))
        # 抽取具备特定关系的命名实体所在窗口的上下文
        rtuple = nltk.sem.rtuple(rel, lcon=True, rcon=True)
        # print(nltk.sem.rtuple(rel, lcon = True, rcon = True))

# 7.7 小结
# * 信息提取系统搜索大量非结构化文本，寻找特定类型的实体和关系，并将它们用来填充有组织的数据库。
#   这些数据库可以用来寻找特定问题的答案
# * 信息提取系统的典型结构以断句开始，然后是分词和词性标注。
#   接下来在产生的数据中搜索特定类型的实体。
#   最后，信息提取系统着眼于文本中提到的相互邻近的实体，并试图确定这些实体之间是否有指定的关系
# * 实体识别通常采用分块器，分割多标识符序列，并且使用适当的实体类型给块加标签。
#   常见的实体类型包括：组织、人员、地点、日期、时间、货币、GPE（地缘政治实体）
# * 利用基于规则的的系统可以构建分块器，NLTK中的RegexpParser类；
#   或者使用机器学习技术，NLTK中的ConsecutiveNPChunker类。
#   词性标记是搜索分块时的重要特征
# * 虽然分块器专门用来建立相对平坦的数据结构，其中任意两个块不允许重叠，但是分块器仍然可以被串联在一起，建立块的嵌套结构
# * 关系抽取可以使用基于规则的系统查找文本中的联结实体和相关词的特定模式，即满足关系要求的实体；
#   也可以使用基于机器学习的系统从训练语料中自动学习这种特定模式，然后依据模式抽取满足关系要求的实体。
