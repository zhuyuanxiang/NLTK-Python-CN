# Ch10 分析句子的含义
# 1）自然语言的含义的表示，以及通过计算机进行处理
# 2）将意思表示无限制的语句集相关联
# 3）连接意思表示与句子的程序来存储信息

# Prover9 和 Mace4 在国外的网站已经没办法下载了，去CSDN吧。
# 国外下载地址：https://www.cs.unm.edu/~mccune/prover9/gui/v05.html
# 国内下载地址：https://download.csdn.net/download/thornbird313/10683390

# 安装以后，需要执行下面的配置命令，就可以调用这两个应用程序进行证明和建模
import os

os.environ.setdefault('PROVER9',
                      'C:\\Program Files (x86)\\Prover9-Mace4\\bin-win32\\')

import nltk
from tools import show_subtitle, show_expr

# 10.1 自然语言理解（NLU）
# 10.1.1 查询数据库
# 意思表示的理念和技术框架
# sql语法的缺陷：
# 1）某些语法公式不符合语法的规则
# 2）把数据库的细节加入了语法公式中
nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')

from nltk import load_parser

cp = load_parser('grammars/book_grammars/sql0.fcfg')
# cp = load_parser('grammars/book_grammars/sql0.fcfg',trace = 3)
query = 'What cities are located in China'
trees = list(cp.parse(query.split()))
answer = trees[0].label()['SEM']
query_sql = ' '.join(answer)
show_subtitle("query")
print(query)
show_subtitle("trees")
for tree in trees:
    print(tree)
show_subtitle("answer")
print(answer)
show_subtitle("query_sql")
print(query_sql)

# NLTK Semantic Interpretation Package, SEM = Semantic
from nltk.sem import chat80

rows = chat80.sql_query('corpora/city_database/city.db', query_sql)
for r in rows:
    print(r[0], end=" ")

rows = chat80.sql_demo()

cp = load_parser('grammars/book_grammars/sql1.fcfg')
query = 'What cities are located in China and have populations above 1,000,000'
trees = list(cp.parse(query.split()))
answer = trees[0].label()['SEM']
query_sql = ' '.join(answer)
show_subtitle("query")
print(query)
show_subtitle("trees")
for tree in trees:
    print(tree)
show_subtitle("answer")
print(answer)
show_subtitle("query_sql")
print(query_sql)

# 10.1.2 自然语言、语义和逻辑
# 语义的基本概念：
# 1）在确定的情况下，陈述句非真即假
# 2）名词短语和专有名词的定义指的世界上的东西
# 语句集W的模型是某种规范化表示，其中W中的所有句子都为真。表示模型的方式通常基于集合论。
# 模型用于评估英语句子的真假，并用这种方式来说明表示意思的一些方法

# 10.2 命题逻辑
# 设计一种逻辑语言的目的是使推理更加明确规范。
# 命题逻辑只表示对应的特定语句连接词的语言结构部分。
# 命题逻辑形式中，连接词的对应形式叫做布尔运算符。
# 命题逻辑的基本表达式是命题符号。通常写作P、Q、R等
# 有了命题符号和布尔运算符，可以建立命题逻辑的规范公式的无限集合（简称公式）
nltk.boolean_ops()

# LogicParser()将逻辑表达式分析成表达式的各种子类
read_expr = nltk.sem.Expression.fromstring
read_expr('-(P&Q)')
read_expr('P&Q')
read_expr('P|(R->Q)')
read_expr('P<->--P')

# 从计算的角度看，逻辑是进行推理的重要工具。从假设一步步捡到结论，被称为推理。
# 有效的论证：所有的前提为真时，结论都为真的论证
read_expr = nltk.sem.Expression.fromstring
SnF = read_expr('SnF')
NotFns = read_expr('-FnS')
R = read_expr('SnF-> -FnS')
prover = nltk.Prover9()
prover.prove(NotFns, [SnF, R])

# 为每个命题符号分配一个值，通过查询确定布尔运算符的含义，并用它们替代这些公式的组成成分的值，来计算合成公式的值
# 配对的链表初始化估值，每个配对由语义符号和语义值组成。
val = nltk.Valuation([('P', True), ('Q', True), ('R', False)])
val['P']

# dom和g参数都被忽略了，后面会用于更加复杂的模型
dom = set()
grammar1 = nltk.Assignment(dom)
model1 = nltk.Model(dom, val)
print('(P&Q)=', model1.evaluate('(P&Q)', grammar1))
print('-(P&Q)=', model1.evaluate('-(P&Q)', grammar1))
print('(P&R)=', model1.evaluate('(P&R)', grammar1))
print('(P|R)=', model1.evaluate('(P|R)', grammar1))

# 10.3 一阶逻辑
# 通过翻译自然语言表达式成为一阶逻辑是计算语义的不错的选择
# 10.3.1 语法
# 一阶逻辑保留了所有命题逻辑的布尔运算符。
# 命题被分析成谓词和参数，接近于自然语言的结构的距离
# 一阶逻辑的标准构造规则的术语：独立变量、独立常量、带不同数量的参数的谓词（一元谓词、二元谓词）
# 表达式被称为非逻辑常量；逻辑觉得在一阶逻辑的每个模型中的解释总是相同的。
# 通过为表达式指定类型可以检查一阶逻辑表达式的语法结构。基本类型：e是实体类型；t是公式类型，即有真值的表达式类型。
# 给定两种基本类型，可以形成函数表达式的复杂类型。
# 信号，作为字典实现与非逻辑变量类型之间的关联。
# 同指称，语义上是等价的。
# 约束，与同指称关系不同的关系。存在量词，全称量词。
read_expr = nltk.sem.Expression.fromstring

expr = read_expr('walk(angus)', type_check=True)
print("expr=", expr)
print("expr.argument=", expr.argument)
print("expr.argument.type=", expr.argument.type)
print("expr.function=", expr.function)
print("expr.function.type=", expr.function.type)

sig = {'walk': '<e,t>'}
expr = read_expr('walk(angus)', signature=sig)
print("expr=", expr)
print("expr.free()=", expr.free())
print("expr.argument=", expr.argument)
print("expr.argument.type=", expr.argument.type)
print("expr.function=", expr.function)
print("expr.function.type=", expr.function.type)

read_expr = nltk.sem.Expression.fromstring
# free() 返回在expr中自由变量的集合
read_expr('dog(cyril)').free()
read_expr('dog(x)').free()
read_expr('own(angus,cyril)').free()
read_expr('exists x.dog(x)').free()
read_expr('((some x. walk(x)) -> sing(x))').free()
read_expr('exists x.own(y,x)').free()

# 10.3.2 一阶定理证明
NotFnS = read_expr('-north_of(f,s)')
SnF = read_expr('north_of(s,f)')
R = read_expr('all x. all y. (north_of(x,y) -> -north_of(y,x))')
prover = nltk.Prover9()
prover.prove(NotFnS, [SnF, R])

FnS = read_expr('north_of(f,s)')
prover.prove(FnS, [SnF, R])

# 10.3.3 一阶逻辑语言总结
# 命题逻辑的语法规则，量词的标准规则，组合成一阶逻辑语法

# 10.3.4 真值模型
# 给定一阶逻辑语言L，L的模型M是一个<D,Val>对，其中D是一个非空集合，称为模型的域；Val是一个函数，称为估值函数。
# NLTk的语义关系可以使用标准的集合论的方法表示：作为元组的集合。
dom = set(['b', 'o', 'c'])
v = """
bertie=>b
olive=>o
cyril=>c
boy=>{b}
girl=>{o}
dog=>{c}
walk=>{o,c}
see=>{(b,o),(c,b),(o,c)}
"""
val = nltk.Valuation.fromstring(v)
print(val)

('o', 'c') in val['see']
('b', ) in val['boy']

# 10.3.5 独立变量和赋值
# 在模型中，上下文对应的使用是为变量赋值。这是一个从独立变量到域中实体的映射。
# 赋值使用Assignment()实现，以论述的模型的域为参数。
grammar1 = nltk.Assignment(dom, [('x', 'o'), ('y', 'c')])
print(grammar1)  # 与逻辑学课本中出现的符号类似
grammar1

# 一阶逻辑的原子公式估值
model1 = nltk.Model(dom, val)
model1.evaluate('see(olive,y)', grammar1)
model1.evaluate('see(olive,cyril)', grammar1)
grammar1['y']
model1.evaluate('see(y,x)', grammar1)
model1.evaluate('see(cyril,olive)', grammar1)

grammar1.purge()  # 清除所有的绑定
grammar1

# 确定模型中公式的真假的一般过程称为模型检查
model1.evaluate('see(olive,y)', grammar1)
model1.evaluate('see(olive,cyril)', grammar1)
model1.evaluate('see(bertie,olive) & boy(bertie) & -walk(bertie)', grammar1)

# 10.3.6 量化
model1.evaluate('exists x. (girl(x) & walk(x))', grammar1)
model1.evaluate('girl(x) & walk(x)', grammar1.add('x', 'o'))

fmla1 = read_expr('girl(x)|boy(x)')
model1.satisfiers(fmla1, 'x', grammar1)

# 前面为真，后面也为真，公式就是真；前面为假，后面无论什么，公式都是真
fmla2 = read_expr('girl(x)->walk(x)')
model1.satisfiers(fmla2, 'x', grammar1)

fmla3 = read_expr('walk(x)->girl(x)')
model1.satisfiers(fmla3, 'x', grammar1)

model1.evaluate('all x.(girl(x) -> walk(x))', grammar1)

# 10.3.7 量词范围歧义：使用两个量词规范化地表示一个语句时，可能会发生歧义
v2 = """
bruce=>b
cyril=>c
elspeth=>e
julia=>j
matthew=>m
person=>{b,e,j,m}
admire=>{(j,b),(b,b),(m,e),(e,m),(c,a)}
"""
val2 = nltk.Valuation.fromstring(v2)
dom2 = val2.domain
model2 = nltk.Model(dom2, val2)
grammar2 = nltk.Assignment(dom2)

fmla4 = read_expr('(person(x)-> exists y.(person(y) & admire(x,y)))')
model2.satisfiers(fmla4, 'x', grammar2)

fmla5 = read_expr('(person(y) & all x.(person(x) -> admire(x,y)))')
model2.satisfiers(fmla5, 'y', grammar2)

fmla6 = read_expr('(person(y) & all x.((x=bruce | x=julia) -> admire(x,y)))')
model2.satisfiers(fmla6, 'y', grammar2)

# 10.3.8 模型的建立
# 模型的建立在给定一些句子集合的条件的基础之上，尝试创建一个新的模型。
# 如果成功，那么说明集合是一致的。
# 调用Mace4模型生成器，将候选的句子集合作为假设，保留目标为未指定
a3 = read_expr('exists x.(man(x) & walk(x))')
c1 = read_expr('mortal(socrates)')
c2 = read_expr('-mortal(socrates)')
mb = nltk.Mace(
    5
)  # The maximum model size that Mace will try before simply returning false.
# a3与c1一致，a3与c2一致。因为Mace成功为a3与c1建立了模型，也为a3与c2建立了模型
# 模型建立器可以作为定理证明器的辅助
print(mb.build_model(None, [a3, c1]))
print(mb.build_model(None, [a3, c2]))
print(mb.build_model(None, [c1, c2]))

a4 = read_expr('exists y.(woman(y) & all x. (man(x) -> love(x,y)))')
a5 = read_expr('man(adam)')
a6 = read_expr('woman(eve)')
grammar3 = read_expr('love(adam,eve)')
mc = nltk.MaceCommand(grammar3, assumptions=[a4, a5, a6])
mc.build_model()
print(mc.valuation)
# C1是一个“Skolem常量”，模型生成器作为存在量词的表示被引入
a7 = read_expr('all x. (man(x) -> -woman(x))')
grammar4 = read_expr('love(adam,eve)')
mc = nltk.MaceCommand(grammar4, assumptions=[a4, a5, a6, a7])
mc.build_model()
print(mc.valuation)

# 以上的语法生成模型，就是通过模型生成来确定语法规则是否完善，如果

# 10.4 英语语句的语义
# 10.4.1 基于特征方法的成分语义学
# 组合原则：整体的含义是部分的含义与它们的句法结合方式的函数。
# 目标是以一种可以与分析过程平滑对接的方式整合语义表达的构建。
# λ 运算：在组装英文句子的意思表示时组合一阶逻辑表达式。

# 10.4.2 λ 运算
# λ 运算符，约束运算符，一阶逻辑量词。
# λ-抽象。表示动词短语（或者无主语从句），作为参数出现在自己的右侧时。
# 开放公式 φ 有自由变量 x，x 抽象为属性表达式 λx.φ——满足 φ 的 x 的属性。
# β-约简：简化语言表示
expr = read_expr(r'\x.(walk(x) & chew_gum(x))')
show_expr(expr)

expr = read_expr(r'\x.(walk(x) & chew_gum(x))(gerald)')
show_expr(expr)

print(read_expr(r'\x. \y. (dog(x) & own(y,x))(cyril)').simplify())
print(read_expr(r'\x \y (dog(x) & own(y,x))(cyril,angus)').simplify())
print(read_expr(r'\x. \y. (dog(x) & own(y,x))(cyril,angus)').simplify())
print(read_expr(r'\x y. (dog(x) & own(y,x))(cyril,angus)').simplify())

# 不可以将一个 λ-抽象作为另一个 λ-抽象的参数
expr = read_expr(r'\y. y(angus))')  # 不可以
expr = read_expr(r'\y. y(angus) (\x.walk(x))')  # 不可以，自由变量y的规定是e类型
expr = read_expr(
    r'\P. P(angus) (\x.walk(x))')  # 可以，命题符号，更高级的类型的变量抽象，如：P、Q作为<e,t>类型的变量
expr = read_expr(r'\y. dog(y) (\x.walk(x))')  # 可以
show_expr(expr)

# 语义等价？
expr1 = read_expr(r'\y.see(y,x)')
expr2 = read_expr(r'\y.see(y,z)')
print(expr1.simplify())
print(expr2.simplify())
expr1.simplify() == expr2.simplify()

# 语义等价？
expr1 = read_expr(r'\P.exists x.P(x)(\y.see(y,x))')
expr2 = read_expr(r'\P.exists x.P(x)(\y.see(y,z))')
print(expr1.simplify())
print(expr2.simplify())
expr1.simplify() == expr2.simplify()

# α-等价，也叫字母变体。重新标记绑定的变量的过程称为α-转换。
expr1 = read_expr('exists x.P(x)')
expr2 = expr1.alpha_convert(nltk.sem.Variable('z'))
print(expr1.simplify())
print(expr2.simplify())
expr1.simplify() == expr2.simplify()

# logic中的β-约简代码自动地重新标记
expr3 = read_expr('\P.(exists x.P(x))(\y.see(y,x))')
print(expr3)
print(expr3.simplify())

# 10.4.3 量化的NP
# 将主语的SEM值作为函数表达式，而不是参数（有时叫做类型提升）
expr1 = read_expr(r'exists x.(dog(x) & bark(x))')
expr2 = read_expr(r'\P.exists x.(dog(x) & P(x))')
print(expr1.simplify())
print(expr2.simplify())
expr1.simplify() == expr2.simplify()

print(expr1.free())
print(expr2.free())
expr1.free() == expr2.free()

expr = read_expr(r'\P.all x.(dog(x) -> P(x))')
expr = read_expr(r'\Q P.exists x.(Q(x) & P(x))')
expr = read_expr(r'\P. P(angus) (\x.walk(x))')
expr = read_expr(r'\P. P(x) (\x.walk(x))')
expr = read_expr(r'\P. P(x) (\y.walk(y))')
expr = read_expr(r'\P.exists x.P(x) (\x.walk(x))')  # 无法正确约简
expr = read_expr(r'\P.exists x.P(x) (\y.walk(y))')  # 无法正确约简
expr = read_expr(r'(\P.exists x.P(x)) (\x.walk(x))')
# 通过print(expr)可以找出程序的理解的结果
expr = read_expr(r'\P.(exists x.P(x)) (\x.walk(x))'
                 )  # print(expr) == (\P.exists x.P(x))(\x.walk(x))
expr = read_expr(r'\P.(exists x.P(x)) (\y.walk(y))')
expr = read_expr(r'exists x.(\P.P(x) (\y.walk(y)))')
expr = read_expr(r'exists x.(dog(x) & \P.P(x) (\x.bark(x)))')
expr = read_expr(r'exists x.(dog(x) & \P.P(x) (\y.bark(y)))')
expr = read_expr(r'(\P Q.exists x.(P(x) & Q(x))) (\x.dog(x))(\x.bark(x))')
expr = read_expr(r'(\P Q.exists x.(Q(x) & P(x))) (\x.dog(x))(\x.bark(x))')
expr = read_expr(r'(\P Q.exists x.(P(x) & Q(x))) (\x.dog(x),\x.bark(x))')
expr = read_expr(r'(\P Q.exists x.(Q(x) & P(x))) (\x.dog(x),\x.bark(x))')
expr = read_expr(r'(\Q P.exists x.(Q(x) & P(x))) (\x.dog(x))(\x.bark(x))')
expr = read_expr(r'(\Q P.exists x.(Q(x) & P(x))) (\x.dog(x),\x.bark(x))')
expr = read_expr(r'exists x.(\Q.Q(x) (\x.dog(x)) & \P.P(x)(\x.bark(x)))')
expr = read_expr(
    r'exists x.(\Q.Q(x) & \P.P(x)(\x.dog(x),\x.bark(x)))')  # 无法正确约简

show_expr(expr)

# 10.4.4 及物动词
expr = read_expr(r'\y.exists x.(dog(x) & chase(y,x))')
expr = read_expr(r'\P.(exists x.(dog(x) & P(x)))(\z.chase(y,z))')
expr = read_expr(r'exists x.(dog(x) & \P.P(x)(\z.chase(y,z)))')
tvp = read_expr(r'\X x.X(\y.chase(x,y))')
tvp = read_expr(
    r'\X (\x.X(\y.chase(x,y)))')  # print(tvp)=>\X x.X(\y.chase(x,y))
expr = tvp
np = read_expr(r'\P.exists x.(dog(x) & P(x))')
np = read_expr(r'(\P.exists x.(dog(x) & P(x)))')
np = read_expr(r'(\P.(exists x.(dog(x) & P(x))))')
expr = np
# nltk.sem.ApplicationExpression() 会为 np 加上括号
vp = nltk.sem.ApplicationExpression(
    tvp, np)  # print(vp)=>(\X x.X(\y.chase(x,y)))(\P.exists x.(dog(x) & P(x)))
expr = vp
expr = read_expr(r'(\X x.X(\y.chase(x,y)))(\P.exists x.(dog(x) & P(x)))')
expr = read_expr(
    r'(\X x.X(\y.chase(x,y)))\P.exists x.(dog(x) & P(x))')  # 不能正确读入
expr = read_expr(
    r'(\P.exists x.(dog(x) & P(x)))(\X x.X(\y.chase(x,y)))')  # 不能正确约简
show_expr(expr)

# simple-sem.fcfg包含了一个用于分析和翻译简单例子的小型规则集合
from nltk import load_parser

parser = load_parser('grammars/book_grammars/simple-sem.fcfg', trace=0)
parser = load_parser('grammars/book_grammars/simple-sem.fcfg', trace=2)
sentence = 'Angus gives a bone to every dog'
tokens = sentence.split()
for tree in parser.parse(tokens):
    print(tree.label()['SEM'])

# interpret_sents() 用于批量地解释输入的语句列表。输出的是语法表达（synrep）和语义表达（semrep）
sents = ['Irene walks', 'Cyril bites an ankle']
grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
for results in nltk.interpret_sents(sents, grammar_file):
    for (synrep, semrep) in results:
        print("语法表达：")
        print(synrep)
        print("语义表达：")
        print(semrep)

# evaluate_sents() 用于批量地评估输入的语句列表，输出的是语法表达（synrep）、语义表达（semrep）和真值(value)
v = """
bertie=>b
olive=>o
cyril=>c
boy=>{b}
girl=>{o}
dog=>{c}
walk=>{o,c}
see=>{(b,o),(c,b),(o,c)}
"""
val = nltk.Valuation.fromstring(v)
grammar = nltk.Assignment(val.domain)
model = nltk.Model(val.domain, val)
sent = 'Cyril sees every boy'
grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
results = nltk.evaluate_sents([sent], grammar_file, model, grammar)[0]
for (synrep, semrep, value) in results:
    print(synrep)
    print(semrep)
    print(value)

# 10.4.5 量词歧义（同一个句子的不同的量词表示，可以跳过）
# 在语义表示与句法分析紧密耦合的前提下，语义中量词的范围也反映了句法分析树中对应的NP的相对范围。
# Cooper存储：是由“核心”语义表示与绑定操作符链表组成的配对。
# S-检索：结合了绑定操作符与核心的操作。
# 建立一个“核心+存储表示”的组合
# nltk.sem.cooper_storage将存储形式的语义表示转换成标准逻辑形式
from nltk.sem import cooper_storage as cs

sentence = 'every girl chases a dog'
trees = cs.parse_with_bindops(sentence,
                              grammar='grammars/book_grammars/storage.fcfg')
semrep = trees[0].label()['SEM']
cs_semrep = cs.CooperStore(semrep)
print(cs_semrep.core)
for bo in cs_semrep.store:
    print(bo)

# 检索与恢复语义表示为标准逻辑形式
cs_semrep.s_retrieve(trace=True)

# 两种标准逻辑形式（存在量词形式、全称量词形式）
for reading in cs_semrep.readings:
    print(reading)

# 10.5 段落（话语）语义
# 段落是语句的序列。组成段落的句子的解释依赖于它前面的句子。
# 照应代词（anaphoric pronouns），即回指代词

# 10.5.1 段落表示理论（Discourse Representation Theory，DRT），即话语表示理论。
# 一阶逻辑中的量化标准方法仅限于单个句子，下面介绍可以扩大到多个句子的方法。
# 段落表示理论的目标是提供处理多个句子以及其他段落特征的语义现象的方法。
# 段落表示结构（Discourse Representation Structure，DRS）根据一个段落指称的列表和一个条件列表表示段落的意思。
# 段落指称是段落中正在讨论的事情，对应于一阶逻辑中的单个变量
# 段落表示结构的条件应用于段落指称，对应于一阶逻辑中的原子开放公式。
read_dexpr = nltk.sem.DrtExpression.fromstring
drs1 = read_dexpr('([x,y],[angus(x),dog(y),own(x,y)])')
show_expr(drs1)

drs1.draw()

# 使用fol()函数，可以把每一个段落表示结构都可以转化为一阶逻辑公式
print(drs1.fol())

# DRT表达式中可以使用DRS连接运算符（“+”）
# DRS连接是指一个单独的DRS包含合并的段落指称和来自多个论证的条件
# DRS连接可以通过α-转换（simplify()）自动完成
drs2 = read_dexpr('([x],[walk(x)])+([y],[run(y)])')
show_expr(drs2)

drs2 = read_dexpr('([x],[walk(x)])+([y],[run(y),see(x,y)])')
show_expr(drs2)
print(drs2.fol())

drs2 = read_dexpr('([x],[walk(x)])+([y],[run(y)])+([z],[see(x,z)])')
show_expr(drs2)
print(drs2.fol())

# 一个DRS可以内嵌入另一个DRS，这是一般量词被处理的方式
drs3 = read_dexpr('([],[(([x],[dog(x)]) -> ([y],[ankle(y),bite(x,y)]))])')
show_expr(drs3)
print(drs3.fol())

# DRT通过链接回指代词和现有的段落指称来解释回指代词。
# DRT设置约束条件使段落指称可以像先行词那样“可访问”，但是并不打算解释如何从候选集合中选出特定的先行词
# resolve_anaphora()将DRS中包括的PRO(x)形式的条件转换为x=[...]形式的条件，其中[...]是一个可能先行词的列表
drs4 = read_dexpr('([x,y],[angus(x),dog(y),own(x,y)])')
drs5 = read_dexpr('([u,z],[PRO(u),irene(z),bite(u,z)])')
drs6 = drs4 + drs5
show_expr(drs6)
print(drs6.simplify().resolve_anaphora())

# 对DRS的处理与 λ-抽象的处理的机制是完全兼容的，因此可以直接基于DRT而不是一阶逻辑建立组合语义表示。
# LogicParser与DrtParser解析的结果似乎是一样的
from nltk import load_parser

parser = load_parser('grammars/book_grammars/drt.fcfg')
parser = load_parser('grammars/book_grammars/drt.fcfg',
                     logic_parser=nltk.sem.drt.DrtParser())
trees = list(parser.parse('Angus owns a dog'.split()))
print(trees[0].label()['SEM'].simplify())

# 10.5.2 段落（话语）处理
# 解释一句话时会利用丰富的上下文背景知识：一部分取决于前面的内容；一部分取决于背景假设。
# DRT提供了将句子的含义集成到前面的段落表示中的理论基础，但是缺乏：没有纳入推理；处理单个句子

# 使用nltk.inference.discourse来完善。
dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
dt.readings()

# 给段落增加句子，使用“consistchk=True”就可以对加入的句子进行一致性检查
dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
show_subtitle("reading sentence")
dt.readings()
show_subtitle("add sentence")
dt.add_sentence('No person dances', consistchk=True)
show_subtitle("reading sentence")
dt.readings()

dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
show_subtitle("reading sentence")
dt.readings()
show_subtitle("add sentence")
dt.add_sentence('No person dances')
show_subtitle("reading sentence")
dt.readings()

# 对于有问题的句子可以收回，“verbose = True”默认就是这个设置，输出撤回句子后的句子列表
dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
show_subtitle("add sentence")
dt.add_sentence('No person dances')
show_subtitle("reading sentence")
dt.readings()
show_subtitle("retract sentence")
dt.retract_sentence('No person dances', verbose=True)
show_subtitle("reading sentence")
dt.readings()

dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
show_subtitle("add sentence")
dt.add_sentence('No person dances')
show_subtitle("reading sentence")
dt.readings()
show_subtitle("retract sentence")
dt.retract_sentence('No person dances')
show_subtitle("reading sentence")
dt.readings()

# 给段落增加句子，使用“informchk=True”就可以对加入的句子进行信息量检查（即是否增加了新的信息量）
dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
show_subtitle("add sentence")
dt.add_sentence('A person dances', informchk=True)
show_subtitle("reading sentence")
dt.readings()

# discourse模型可以适应语义歧义，筛选出不可接受的读法。
# Glue语义模型被配置为使用覆盖面广泛的Malt依存关系分析器，输入的句子必须已经完成了分词和标注。
# MaltParser()需要去 http://www.maltparser.org/mco/mco.html 下载 MaltParser，
# 然后解压缩到合适的目录下，使用 parser_dirname 来设置目录
from nltk.tag import RegexpTagger

tagger = RegexpTagger([('^(chases|runs)$', 'VB'), ('^(a)$', 'ex_quant'),
                       ('^(every)$', 'univ_quant'), ('^(dog|boy)$', 'NN'),
                       ('^(He)$', 'PRP')])
depparser = nltk.MaltParser(
    tagger=tagger,
    parser_dirname='D:\\Users\\zhuyuanxiang\\Library\\maltparser')

depparser = nltk.MaltParser(
    tagger=tagger,
    parser_dirname='D:\\Users\\zhuyuanxiang\\Library\\maltparser-1.9.2')
rc = nltk.DrtGlueReadingCommand(depparser=depparser)
dt = nltk.DiscourseTester(['Every dog chases a boy', 'He runs'], rc)
dt.readings()

# TypeError: 'RegexpTagger' object is not callable
# 估计是版本不匹配造成的

import nltk

pattern = [(r'(March)$', 'MAR')]
tagger = nltk.RegexpTagger(pattern)
print(tagger.tag('He was born in March 1991'))
print(tagger.tag(nltk.word_tokenize('He was born in March 1991')))

# 下面是知乎上给出的修改建议，测试了依然不行。
# 具体可参考 https://www.zhihu.com/people/meng-hui-wei-lai-de-colin/activities
tagger = RegexpTagger([('^(chases|runs)$', 'VB'), ('^(a)$', 'ex_quant'),
                       ('^(every)$', 'univ_quant'), ('^(dog|boy)$', 'NN'),
                       ('^(He)$', 'PRP')])
depparser = nltk.MaltParser(
    tagger=tagger.tag,
    parser_dirname='D:\\Users\\zhuyuanxiang\\Library\\maltparser')
rc = nltk.DrtGlueReadingCommand(depparser=depparser)
dt = nltk.DiscourseTester(
    [sent.split() for sent in ['Every dog chases a boy']], reading_command=rc)
dt.readings()

[sent.split() for sent in ['Every dog chases a boy', 'He runs']]
[nltk.word_tokenize('Every dog chases a boy'), nltk.word_tokenize('He runs')]

# 10.6 小结
# - 一阶逻辑是一种适合在计算环境中表示自然语言含义的语言。
#     - 可以表示自然语言含义的许多方面，
#     - 还可以使用一阶逻辑失量的高效的定理证明器。
# - 将自然语言句子翻译成一阶逻辑的时候，可以通过检查一阶公式模型表述这些句子的真值条件
# - 一阶逻辑的 λ-运算，可以表示成分组合的含义
# - λ-运算中的 β-约简。
#     - 在语义上，与函数传递参数对应。
#     - 在句法上，将被函数表达式中的 λ 绑定的变量替换为函数应用中表达式提供的参数。
# - 构建模型的关键部分在于建立估值，为非逻辑常量分配解释。这些非逻辑常量可以解释为n元谓词或者独立常量
# - 开放表达式是包含一个或者多个自变量的表达式。开放表达式只在它的自变量被赋值时才能获得解释。
# - 量词的解释是对于具有变量x的公式φ[x]，构建个体的集合，赋值g分配它们作为x的值使φ[x]为真。然后量词对这个集合加以约束。
# - 封闭的表达式是没有自由变量的表达式，即变量都是被绑定的。封闭的表达式的真假取决于所有变量的赋值。
# - 如果两个公式只是由绑定操作符（即 λ 或者 量词）绑定的变量的标签不同，那么它们是 α-等价的。重新标记公式中的绑定变量的过程叫做 α-转换。
# - 给定有两个嵌套量词Q1和Q2的公式，最外层的量词Q1有比较宽的范围（或者范围超过Q2）。英语句子往往由于它们包含的量词的范围而产生歧义。
# - 在基于特征的语法中英语句子可以通过将SEM作为特征与语义表达关系。一个复杂表达式的SEM值通常包括成分表达式的SEM值的函数应用。
