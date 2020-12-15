# Ch11 语言数据管理
# 1）设计新的语言资源，需要确保覆盖面、平衡及对文档的广泛支持
# 2）将现有数据转换成合适的格式用于分析
# 3）发布已经创建的资源，方便人们查找和使用

# 11.1 语料库结构：安全研究
# TIMIT语料库是第一个广泛发布的已经标注的语音数据库，为获取声学——语音知识提供数据，支持自动语音识别系统的开发和评估
# 11.1.1 TIMIT结构
import nltk

for i, fileid in enumerate(nltk.corpus.timit.fileids()):
    if i < 10:
        print(i, ')', fileid)

# TIMIT记录了音标和单词，通过phones()可以查看音标
phonetic = nltk.corpus.timit.phones('dr1-fvmh0/sa1')
print(phonetic)
nltk.corpus.timit.word_times('dr1-fvmh0/sa1')

# TIMIT提供了规范的发音
timitdict = nltk.corpus.timit.transcription_dict()
print(timitdict['greasy'] + timitdict['wash'] + timitdict['water'])
print(phonetic[17:30])

# 说话人的相关信息
nltk.corpus.timit.spkrinfo('dr1-fvmh0')

# 11.1.2 TIMIT的主要设计特点
# TIMIT语料库设计中的主要特点：
# 1） 语料库中包含了语音和字形两个标注层
# 2） 语料库在多个维度的变化与方言地区和二元音覆盖范围之间取得了平衡
# 3） 语料库区分了作为录音来捕捉和作为标注来捕捉的原始语言学事件
# 4） 语料库的层次结构
# 5） 语料库中除了包含了语音数据，还包含了词汇和文字数据

# 11.1.3 TIMIT的基本数据类型
# TIMI包含了两种基本数据类型：词典和文本
# 词典资源使用记录结构表示，即一个关键字加一个或者多个字段。
# 词典资源 可以是一个传统字典或者比较词表，也可以是一个短语词典，其中的关键字是一个短语而不是一个词
# 词典还包括了结构化数据的记录，可以通过对应主题的非关键字字段来查找条目
# 可以通过构造特征的表格（称为范例）来进行对比和说明系统性的变化
# 说话者表也是一种词典资源

# 在最抽象的层面上，文本表示真实的或者虚构的讲话事件，这个事件的时间过程也在文本本身存在。
# 文本可以是一个小单位，如一个词或者一个句子；也可以是一个完整的叙述或者对话。

# 11.2 语料库的生命周期
# 11.2.1 创建语料库的3种方案
# 1） “领域语言学”模式，即来自会话的材料在被收集的同时就被分析。
# 2） 实验研究模式，从人类中收集主，然后分析来评估一个假设或者开发一种技术。这类数据库是“共同任务”的科研管理方法的基础
# 3） 特定的语言收集“参考语料”。

# 11.2.2 质量控制
# 标注指南确定任务并且记录标记约定。
# Kappa系数测量两个人判断类别，修正预期期望的一致性。
# windowdiff()评估两个分割的一致性得分。3是窗口的大小
s1 = '00000010000000001000000'
s2 = '00000001000000010000000'
s3 = '00010000000000000001000'
print(nltk.windowdiff(s1, s1, 3))
print(nltk.windowdiff(s1, s2, 3))
print(nltk.windowdiff(s2, s3, 3))

print(nltk.windowdiff(s1, s2, 6))
print(nltk.windowdiff(s2, s3, 6))

print(nltk.windowdiff(s1, s2, 1))
print(nltk.windowdiff(s2, s3, 1))

# 11.2.3 维护与演变
# 发布原始语料库需要一个能够识别其中任何一部分的规范。
# 每个句子、树或者词条都有全局唯一的标识符
# 每个标识符、节点或者字段（分别）都有一个相对领衔
# 标注包括分割，可以使用规范的标识符引用原始材料。
# 新的标注可以与原始材料独立分布，同一个来源的多个独立标注可以对比和更新而不影响原始材料。

# 11.3 数据采集
# 11.3.1 从网络上获取数据
# RSS订阅、搜索引擎的结果、发布的网页。

# 11.3.2 从文本文件中获取数据
# 从Word文档中获取数据
# 将Word文档转存为HTML文档

# 从HTML文档中获取数据
# 1) 从Word转存的HTML文件与书上的格式不同
# 2）取出的结果也与书上的不同

import re

legal_pos = {'n', 'v.t.', 'v.i.', 'adj', 'det'}
# pattern = re.compile(r"'font-size:11.0pt'")
# pattern = re.compile(r">([a-z.]+)<")
pattern = re.compile(r"'font-size:11.0pt'>([a-z.]+)<")
document = open('dict.htm').read()
print(document)
used_pos = set(re.findall(pattern, document))
print(list(used_pos))
# 非法词性的集合
illegal_pos = used_pos.difference(legal_pos)
print(list(illegal_pos))

# 将HTML文档转存为CSV文档
# 不能正确地显示结果
from bs4 import BeautifulSoup


def lexical_data(html_file, encoding='utf-8'):
    SEP = '_ENTRY'
    html = open(html_file, encoding=encoding).read()
    print('html1:', html)
    html = re.sub(r'<p', SEP + '<p', html)
    print('html2:', html)
    text = BeautifulSoup(html, 'html.parser').get_text()
    print('text1:', text)
    text = ' '.join(text.split())
    print('text2:', text)
    for entry in text.split(SEP):
        print('entry:', entry)
        if entry.count(' ') > 2:
            yield entry.split(' ', 3)


import csv

dict_csv = open('dict.csv', 'w')
writer = csv.writer(dict_csv)
writer.writerows(lexical_data('dict.htm'))
dict_csv.close()

# 11.3.3 从电子表格和数据库中获取数据
import csv

# 注意删除 dict.csv 中多余的空行
dict_csv = open('dict.csv')
lexicon = csv.reader(dict_csv)
for (lexeme, _, _, defn) in lexicon:
    print(lexeme,defn)
pairs = [(lexeme, defn) for (lexeme, _, _, defn) in lexicon]
lexemes, defns = zip(*pairs)
defn_words = set(w for defn in defns for w in defn.split())
sorted(defn_words.difference(lexemes))
dict_csv.close()

# 11.3.4 数据格式的转换
idx = nltk.Index((defn_word, lexeme)
                 for (lexeme, defn) in pairs
                 for defn_word in nltk.word_tokenize(defn)
                 if len(defn_word) > 3)
with open('dict.idx', 'w') as idx_file:
    for word in sorted(idx):
        idx_words = ','.join(idx[word])
        idx_line = '{}: {}'.format(word, idx_words)
        print(idx_line)

# 11.3.5 选择需要保留的标注层
# 常用的标注层：
# - 分词：文本的书写形式不能明确地识别它的标识符。分词和规范化的版本作为常规的正式版本的补充
# - 断句：因为断句的困难，因此语料库为断句提供明确的标注
# - 分段：明确注明段落和其他结构元素（标题、章节等）
# - 词性：文档中的每个单词的词类
# - 句法结构：一个树状结构显示一个句子的组成结构
# - 浅层语义：命名实体和共指标注，语义角色标签
# - 对话与段落：对话行为标记，修辞结构
# 内联标注：通过插入带有标注信息的特殊符号或者控制序列修改原始文档
# 对峙标注：不修改原始文档，而是创建一个新的文档，通过使用指针引用原始文档来增加标注信息

# 11.3.6 标准和工具
# 共同的接口：抽象数据类型、面向对象设计、三层结构

# 11.3.7 处理濒危语言时的特征考虑
# SIL的自由软件Toolbox和Filedwords对文本和词汇的创建集成提供了很好的支持
# 使用词义范畴标注词项，允许通过语义范畴或者注释查找
# 去除单词中的元音、缩写和重复字母
mappings = [('ph', 'f'), ('ght', 't'), ('^kn', 'n'), ('qu', 'kw'), ('[aeiou]+', 'a'), (r'(.)\1', r'\1')]


def signature(word):
    for patt, repl in mappings:
        word = re.sub(patt, repl, word)
    pieces = re.findall('[^aeiou]+', word)
    return ''.join(char for piece in pieces for char in sorted(piece))[:8]


signature('illefent')
signature('ebsekwieous')
signature('nuculerr')

# 寻找相同的编码的单词
signatures = nltk.Index((signature(w), w) for w in nltk.corpus.words.words())
signatures[signature('nuculerr')]

# 寻找相同的编码的单词
def rank(word, wordlist):
    ranked = sorted((nltk.edit_distance(word, w), w) for w in wordlist)
    return [word for (_, word) in ranked]


def fuzzy_spell(word):
    sig = signature(word)
    if sig in signatures:
        return rank(word, signatures[sig])
    else:
        return []


fuzzy_spell('illefent')
fuzzy_spell('ebsekwieous')
fuzzy_spell('nucular')

# 11.4 使用XML
# XML(The Extensible Markup Language, 可扩展标记语言)为设计特定领域的标记语言提供了一个框架。
# 用于表示已经被标注的文本和词汇资源
# XML允许创建自己的标签；允许创建的数据而不必事先指定其结构；允许有可选的、可重复的元素。

# 11.4.1 在语言结构中使用XML
# 在结构的XML中，在嵌套的同一级别中所有的开始标签必须结束标记（即XML文档必须是格式良好的树）。
# XML允许使用重复的元素
# XML使用“架构（scema）”限制一个XML文件的格式，是一种类似于上下文无关方法的声明。

# 11.4.2 XML的作用
# XML提供了一个格式方便和用途广泛的工具

# 11.4.3 ElementTree接口
# Python的ElementTree模型提供了一种方便的方式用于访问存储在XML文件中的数据。
merchant_file = nltk.data.find('corpora/shakespeare/merchant.xml')
raw = open(merchant_file).read()
print(raw[:163])
print(raw[1789:2006])

from xml.etree.ElementTree import ElementTree

merchant = ElementTree().parse(merchant_file)
merchant
merchant[0]
merchant[0].text
merchant.getchildren()
merchant[-2][0].text
merchant[-2][1]
merchant[-2][1][0].text
merchant[-2][1][54]
merchant[-2][1][54][0]
merchant[-2][1][54][0].text
merchant[-2][1][54][1]
merchant[-2][1][54][1].text

for i, act in enumerate(merchant.findall('ACT')):
    for j, scene in enumerate(act.findall('SCENE')):
        for k, speech in enumerate(scene.findall('SPEECH')):
            for line in speech.findall('LINE'):
                if 'music' in str(line.text):
                    print('Act %d Scene %d Speech %d: %s' % (i + 1, j + 1, k + 1, line.text))

from collections import Counter

speaker_seq = [s.text for s in merchant.findall('ACT/SCENE/SPEECH/SPEAKER')]
speaker_freq = Counter(speaker_seq)
top5 = speaker_freq.most_common(5)
print(top5)

# 由于有23个演员，只选择前五位角色之间相互对话
from collections import defaultdict

abbreviate = defaultdict(lambda: 'OTH')
for speaker, _ in top5:
    abbreviate[speaker] = speaker[:4]
speaker_seq2 = [abbreviate[speaker] for speaker in speaker_seq]
cfd = nltk.ConditionalFreqDist(nltk.bigrams(speaker_seq2))
cfd.tabulate()

# 11.4.4 使用ElementTree访问Toolbox的数据
from nltk.corpus import toolbox

# 访问lexicon对象的内容的两种方法
# 1） 通过索引
# 索引访问：lexicon[3]返回3号条目（从0开始算起的第4个条目），lexicon[3][0]返回它的第一个字段
lexicon = toolbox.xml('rotokas.dic')
lexicon[3][0]
lexicon[3][0].tag
lexicon[3][0].text

# 2） 通过路径
# 路径访问：'record/lx'的所有匹配，并且访问该元素的文本内容，将其规范化为小写
[lexeme.text.lower() for lexeme in lexicon.findall('record/lx')]

# Toolbox数据是XML格式。
import sys
from nltk.util import elementtree_indent
from xml.etree.ElementTree import ElementTree

elementtree_indent(lexicon)
tree = ElementTree(lexicon[3])
tree.write(sys.stdout, encoding='unicode')

# 11.4.5 格式化条目
# 将数据转换为HTML格式输出
html = "<table>\n"
for entry in lexicon[70:80]:
    lx = entry.findtext('lx')
    ps = entry.findtext('ps')
    ge = entry.findtext('ge')
    html += ' <tr><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (lx, ps, ge)
html += '</table>'
print(html)

# 11.5 使用Toolbox数据
from nltk.corpus import toolbox

lexicon = toolbox.xml('rotokas.dic')
print(sum(len(entry) for entry in lexicon) / len(lexicon))

# 11.5.1 为每个条目添加字段
# Ex11-2 为词汇条目添加新的cv字段
from xml.etree.ElementTree import SubElement


def cv(s):
    s = s.lower()
    s = re.sub(r'[^a-z]', r'_', s)
    s = re.sub(r'[aeiou]', r'V', s)
    s = re.sub(r'[^V_]', r'C', s)
    return (s)


def add_cv_field(entry):
    for field in entry:
        if field.tag == 'lx':
            cv_field = SubElement(entry, 'cv')
            cv_field.text = cv(field.text)


lexicon = toolbox.xml('rotokas.dic')
add_cv_field(lexicon[53])
print(nltk.toolbox.to_sfm_string(lexicon[53]))

# 11.5.2 验证Toolbox词汇
# 使用 Counter() 函数快速寻找频率异常的字段序列
from collections import Counter

field_sequences = Counter(':'.join(field.tag for field in entry) for entry in lexicon)
print(field_sequences.most_common())

# Ex11-3 使用上下文无关语法验证Toolbox中的条目
# 基于Ch8介绍的CFG格式。
# 语法模型隐含了Toolbox条目的嵌套结构，建立树状结构，树的叶子是单独的字段名
# 遍历条目并且报告它们与语法的一致性。‘+’表示被语法接受的；‘-’表示不被语法接受的
grammar = nltk.CFG.fromstring('''
S -> Head PS Glosses Comment Date Sem_Field Examples
Head -> Lexeme Root
Lexeme -> "lx"
Root -> "rt" |
PS -> "ps"
Glosses -> Gloss Glosses |
Gloss -> "ge" | "tkp" | "eng"
Date -> "dt"
Sem_Field -> "sf"
Examples -> Example Ex_Pidgin Ex_English Examples |
Example -> "ex"
Ex_Pidgin -> "xp"
Ex_English -> "xe"
Comment -> "cmt" | "nt" |
''')


def validate_lexicon(grammar, lexicon, ignored_tags):
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for entry in lexicon:
        marker_list = [field.tag for field in entry if field.tag not in ignored_tags]
        if list(rd_parser.parse(marker_list)):
            print('+', ':'.join(marker_list))
        else:
            print('-', ':'.join(marker_list))


lexicon = toolbox.xml('rotokas.dic')[10:20]
ignored_tags = ['arg', 'dcsv', 'pt', 'vx']
validate_lexicon(grammar, lexicon, ignored_tags)

# Ex11-4 为Toolbox词典分块：此块语法描述了一种中国方言的词汇条目结构
# 使用块分析器，能够识别局部结构并且报告已经确定的局部结构
grammar = r"""
lexfunc: {<lf>(<lv><ln|le>*)*}
example: {<rf|xv><xn|xe>*}
sense:   {<sn><ps><pn|gv|dv|gn|gp|dn|rn|ge|de|re>*<example>*<lexfunc>*}
record:  {<lx><hm><sense>+<dt>}
"""

from xml.etree.ElementTree import ElementTree
from nltk.toolbox import ToolboxData

db = ToolboxData()
db.open(nltk.data.find('corpora/toolbox/iu_mien_samp.db'))
# db.parse()解析不了
lexicon = db.parse(grammar, encoding='utf8')
tree = ElementTree(lexicon)
with open('iu_mien_samp.xml', 'wb') as output:
    tree.write(output)

# 11.6 使用OLAC元数据描述语言资源
# NLP社区成员共同使用的具有很高精度和召回率的语言资源，已经提供的方法是元数据聚焦

# 11.6.1 什么是元数据？
# “元数据”就是关于数据的结构化数据。是对象或者资源的描述信息。
# 都柏林核心数据（Dublin Core Metadata）由15个元数据元素组成，每个元素都是可选的和可重复的。
# 标题、创建者、主题、描述、发布者、参与者、日期、类型、格式、标识符、来源、语言、关系、覆盖范围和版权
# 开放档案倡议（Open Archives Initiative，OAI）提供了跨越数字化的学术资料库的共同框架，不考虑资源的类型

# 11.6.2 开放语言档案社区（Open Language Archives Community，OLAC）
# 开放语言档案社区正在一种国际性的伙伴关系，这种伙伴关系是创建世界性语言资源的虚拟图书馆的机构和个人
# 1） 制定目前最好的关于语言资源的数字归档实施的共识
# 2） 开发、存储和访问这些资源的互操作信息库和服务的网络
# OLAC元数据是描述语言资源的标准。确保跨库描述的统一性。描述物理和数字格式的数据和工具。添加了语言资源的基本属性。

# 11.6.3 发布语言资源
# 社区成员可以上传语料库和模型来进行发布

# 11.7 小结
# - 语料库中基本数据类型是已经标注的文本和词汇。
#     - 文本有时间结构
#     - 词汇有记录结构
# - 语料库的生命周期，包括：数据收集、标注、质量控制及发布。
# - 语料库开发包括捕捉语言使用的代表性的样本与使用任何一个来源或者文体都有足够的材料之间的平衡；增加变量的维度通常由于资源的限制而不可行
# - XML提供了一种有用的语言数据的存储和交换形式
# - Toolbox格式被使用在语言记录项目中，可以编写程序来支持Toolbox文件的维护，并将它们转换成XML
# - 开放语言社区（OLAC）提供了用于记录和发现语言资源的基础设施
