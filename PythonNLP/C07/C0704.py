# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0704.py
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

# 7.4. 语言结构中的递归
# 7.4.1 使用级联分块器构建嵌套的结构
# Ex7-6 分块器，处理NP（名词短语）PP（介绍短语）VP（动词短语）和$（句子的模式）
grammar = r'''
NP: {<DT|JJ|NN.*>+}             # Chunk sequences of DT, JJ, NN
PP: {<IN><NP>}                  # Chunk prepositions followed by NP
VP: {<VB.*><NP|PP|CLAUSE>+$}    # Chunk verbs and their arguments
CLAUSE: {<NP><VP>}              # Chunk NP, VP
'''

sentence = [("Mary", "NN"),
            ("saw", "VBD"),
            ("the", "DT"),
            ("cat", "NN"),
            ("sit", "VB"),
            ("on", "IN"),
            ("the", "DT"),
            ("mat", "NN")]

sentence = [("John", "NNP"),
            ("thinks", "VBZ"),
            ("Mary", "NN"),
            ("saw", "VBD"),
            ("the", "DT"),
            ("cat", "NN"),
            ("sit", "VB"),
            ("on", "IN"),
            ("the", "DT"),
            ("mat", "NN")]

cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

# 对于有深度的语法结构，一次剖析不能解决问题，使用loop设置循环剖析的次数
cp = nltk.RegexpParser(grammar, loop=2)
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar, loop=3)
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar, loop=4)
parsed_sent = cp.parse(sentence)
print(parsed_sent)
parsed_sent.draw()

# 虽然级联过程可以创建深层结构，但是创建与调试过程非常困难，并且只能产生固定深度的树状图，仍然属于不完整的句法分析
# 因此，全面的剖析才更有效（Ref：Ch8）

# 7.4.2. 树状图：一组相互连接的加标签的节点，从一个特殊的根节点沿一条唯一的路径到达每个节点。
# 创建树状图
tree1 = nltk.Tree('NP', ['Alice'])
print(tree1)

tree2 = nltk.Tree('NP', ['the', 'rabbit'])
print(tree2)

# 合并成大的树状图
tree3 = nltk.Tree('VP', ['chased', tree2])
print(tree3)

tree4 = nltk.Tree('S', [tree1, tree3])
print(tree4)

# 访问树状图的对象
print("tree4= ", tree4)
print("tree4[0]= ", tree4[0])
print("tree4[1]= ", tree4[1])
print("tree4[1][1]= ", tree4[1][1])
print("tree4[1][1][0]= ", tree4[1][1][0])

# 调用树状图的函数
print("tree4.label()= ", tree4.label())
print("tree4.leaves()= ", tree4.leaves())
print("tree4[1].label()= ", tree4[1].label())
print("tree4[1].leaves()= ", tree4[1].leaves())
print("tree4[1][1].label()= ", tree4[1][1].label())
print("tree4[1][1].leaves()= ", tree4[1][1].leaves())
print("tree4[1][1][0].label()= ", tree4[1][1][0].label())
print("tree4[1][1][0].leaves()= ", tree4[1][1][0].leaves())

tree4.draw()


# 7.4.3 树的遍历
# Ex7-7 递归函数遍历树状图
def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t, end=' ')
    else:
        print('(', t.label(), end=' ')
        for child in t:
            traverse(child)
        print(')', end=' ')


# 不能使用Tree()函数直接基于字符串生成树了。
t = nltk.Tree.fromstring('(S (NP Alice) (VP chased (NP the rabbit)))')
print(t)
traverse(t)
t.draw()
t = nltk.Tree.fromstring(tree4.__str__())
traverse(tree4)
