import nltk

# Chap9 建立基于特征的文法（语法）
# 目标：
# 1）怎样用特征扩展无关上下文文法的框架，以获得对文法类别和产生式的更细粒度的控制？
# 2）特征结构的主要形式化发生是什么？如何使用它们来计算？
# 3）基于特征的文法能够获得哪些语言模式和文法结构？

# 9.1 语法特征
# 在基于规则的上下文语法中，特征——值偶对被称为特征结构。
# 字典存储特征以及特征的值
# 'CAT' 表示语法类别；'ORTH'：表示正字法（正词法，拼写规则）
# 'REF' 表示 'kim' 的指示物；'REL'：表示 'chase' 表示的关系
# 'AGT' 表示施事（agent）角色；'PAT'： 表示受事（patient）角色
kim = {'CAT': 'NP', 'ORTH': 'Kim', 'REF': 'k'}
chase = {'CAT': 'V', 'ORTH': 'chased', 'REL': 'chase'}
chase['AGT'] = 'sbj'  # 'sbj'（主语）作为占位符
chase['PAT'] = 'obj'  # 'obj'（宾语）作为占位符
lee = {'CAT': 'NP', 'ORTH': 'Lee', 'REF': 'l'}

sent = "Kim chased Lee"
tokens = sent.split()


def lex2fs(word):
    for fs in [kim, lee, chase]:
        if fs['ORTH'] == word:
            return fs


subj, verb, obj = lex2fs(tokens[0]), lex2fs(tokens[1]), lex2fs(tokens[2])

verb['AGT'] = subj['REF']  # agent of 'chase' is Kim
verb['PAT'] = obj['REF']  # patient of 'chase' is Lee
for k in ['ORTH', 'REL', 'AGT', 'PAT']:  # check featstruct of 'chase'
    print("%-5s => %s" % (k, verb[k]))

# 'SRC'：表示源事（source）的角色；'EXP'：表示体验者（experiencer）的角色
surprise = {'CAT': 'V', 'ORTH': 'surprised', 'REL': 'surprise', 'SRC': 'SBJ', 'EXP': 'obj'}

# 特征结构是非常强大的，特征的额外表现力（Ref：Sec 9.3）开辟了用于描述语言结构复杂性的可能。

# 9.1.1 句法协议
# 协议（agreement）：动词的形态属性和主语名词短语的句法属性一起变化的过程。
# 表9-1 英语规则动词的协议范式

# 9.1.2 使用属性和约束
# Ex9-1 基于特征的语法的例子
nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')

# Ex9-2 跳跃基于特征的图表分析器
tokens = 'Kim likes children'.split()
from nltk import load_parser

cp = load_parser('grammars/book_grammars/feat0.fcfg', trace = 2)
for tree in cp.parse(tokens):
    print(tree)

# 9.1.3 术语
# 简单的值通常称为原子。
#   原子值的一种特殊情况是布尔值。
# AGR是一个复杂值。
# 属性——值矩阵（Attribute-Value Matrix，AVM）

# 9.2 处理特征结构
# 特征结构的构建；两个不同特征结构的统一（合一）运算。
fs1 = nltk.FeatStruct(TENSE = 'past', NUM = 'sg')
print(fs1)

fs1 = nltk.FeatStruct(PER = 3, NUM = 'pl', GND = 'fem')
print(fs1['GND'])
fs1['CASE'] = 'acc'
print(fs1)

fs2 = nltk.FeatStruct(POS = 'N', AGR = fs1)
print(fs2)
print(fs2['AGR'])

print(nltk.FeatStruct("[POS='N',AGR=[PER=3, NUM='pl', GND='fem']]"))

# 特征结构也可以用来表示其他数据
print(nltk.FeatStruct(NAME = 'Lee', TELNO = '13918181818', AGE = 33))

# 特征结构也可以使用有向无环图（Directed Acyclic Graph，DAG）来表示，相当于前面的AVM
# DAG可以使用结构共享或者重入来表示两条路径具有相同的值，即它们是等价的。

# 括号()里面的整数称为标记或者同指标志（coindex）。
print(nltk.FeatStruct("""[Name='Lee', ADDRESS=(1)[NUMBER=74,STREET='rue Pascal'],SPOUSE=[NAME='Kim',ADDRESS->(1)]]"""))

# 9.2.1 包含（蕴涵） 和 统一（合一） （Ref：《自然语言处理综论》Ch15）
# 一般的特征结构包含（蕴涵）特殊的特征结构
# 合并两个特征结构的信息称为统一（合一），统一（合一）运算是对称的。
# 合一的相容运算
fs1 = nltk.FeatStruct(NUMBER = 74, STREET = 'rule Pascal')
fs2 = nltk.FeatStruct(CITY = 'Paris')
print(fs1.unify(fs2))
print(fs2.unify(fs1))
print(fs1.unify(fs2) == fs2.unify(fs1))

# 合一的失败运算
fs0 = nltk.FeatStruct(A = 'a')
fs1 = nltk.FeatStruct(A = 'b')
fs2 = fs0.unify(fs1)
print(fs2)


fs0 = nltk.FeatStruct("[SPOUSE=[ADDRESS=[CITY=Paris]]]")
# 无同指标志的特征结构的相容运算
fs1 = nltk.FeatStruct("""[Name='Lee', 
ADDRESS=[NUMBER=74,STREET='rue Pascal'],
SPOUSE=[NAME='Kim',
ADDRESS=[NUMBER=74,STREET='rue Pascal']]]""")
print(fs0.unify(fs1))

# 有同指标志的特征结构的相容运算
fs2 = nltk.FeatStruct("""[Name='Lee', 
ADDRESS=(1)[NUMBER=74,STREET='rue Pascal'],
SPOUSE=[NAME='Kim',
ADDRESS->(1)]]""")
print(fs0.unify(fs2))

# 使用变量?x表示的特征结构的相容运算
fs1 = nltk.FeatStruct("[ADDRESS1=[NUMBER=74, STREET='rue Pascal'], ADDRESS4=[NAME='Lee']]")
fs2 = nltk.FeatStruct("[ADDRESS1=?x, ADDRESS2=?x, ADDRESS3=?y, ADDRESS4=?y]")
print(fs2)
print(fs2.unify(fs1))

# 9.3 扩展基于特征的文法（语法）
# 9.3.1 子类别（次范畴化）
# 广义短语结构语法（Generalized Phrase Structure Grammar，GPSG），
# 允许词汇类别支持SUBCAT特征（表明项目所属的子类别）

# 9.3.2 回顾核心词概念

# 9.3.3 助动词和倒装

# 9.3.4 无限制依赖成分
# 具有倒装从句和长距离依赖的产生式的语法，使用斜线类别
nltk.data.show_cfg('grammars/book_grammars/feat1.fcfg')
tokens = 'who do you claim that you like'.split()
from nltk import load_parser

cp = load_parser('grammars/book_grammars/feat1.fcfg')
tree=None
for tree in cp.parse(tokens):
    print(tree)
tree.draw()

tokens = 'you claim that you like cats'.split()
for tree in cp.parse(tokens):
    print(tree)
tree.draw()

tokens = 'rarely do you sing'.split()
for tree in cp.parse(tokens):
    print(tree)
tree.draw()

# 9.3.5 德语中的格和性别
# Ex9-4 基于特征的语法的例子（表示带格的协议的相互作用）
nltk.data.show_cfg('grammars/book_grammars/german.fcfg')

tokens = 'ich folge den Katzen'.split()
cp = load_parser('grammars/book_grammars/german.fcfg')
for tree in cp.parse(tokens):
    print(tree)
tree.draw()

tokens='ich folge den Katze'.split()
cp = load_parser('grammars/book_grammars/german.fcfg',trace=2)
for tree in cp.parse(tokens):
    print(tree)
tree.draw()

# 9.4 小结
# * 上下文无关语法的传统分类是原子符号。特征结构的重要作用之一是捕捉精细的区分，否则将需要数量翻倍的原子类别
# * 通过使用特征值的变量，可以表达出语法产生式中的限制，使得不同的特征规格之间相互依赖
# * 在词汇层面指定固定的特征值，并且限制短语中的特征值，使其与“孩子”的对应值相统一（？）
# * 特征值可以是原子的，也可以是复杂的。原子值的特定类别是布尔值
# * 两个特征可以共享一个值（原子的或者复杂的）。具有共享值的结构被称为重入。共享值被表示AVM中的数字索引（或者标记）
# * 特征结构中的路径是特征元组，对应着从图底部开始的弧序列上的标签。
# * 如果两条路径共享一个值，那么这两条路径是等价的。
# * 包含的特征结构是偏序的。特征结构A蕴涵特征结构B，说明特征结构A更加一般，特征结构B更加特征
# * 如果统一（合一）运算在特征结构中指定了一条路径，那么同时指定了所有与这条路径等价的其他路径。
# * 使用特征结构对大量的语言学现象进行简洁的分析，包括：
#       * 动词子类别
#       * 倒装结构
#       * 无限制依赖结构
#       * 格支配

# 9.5 深入阅读
# 理论语言学中使用特征是捕捉语音的音素特征
# 计算语言学提出了语言功能可以被属性——值结构统一捕获
# 词汇功能语法表示语法关系和与成分结构短语关联的谓词参数结构