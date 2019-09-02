import re

import nltk

# 1) 从非结构化文本中提取结构化数据
# 2） 识别一个文本中描述的实体和关系
# 3） 使用语料库来训练和评估模型
# 1. 信息提取
# 1） 从结构化数据中提取信息
# 2） 从非结构化文本中提取信息
#   * 建立一个非常一般的含义
#   * 查找文本中具体的各种信息
#       * 将非结构化数据转换成结构化数据
#       * 使用查询工具从文本中提取信息
# 1.1 信息提取结构
# P283 图7-1
# ｛原始文本（一串）｝→断句→｛句子（字符串列表）｝→分词→｛句子分词｝→词性标注→｛句子词性标注｝→实体识别→｛句子分块｝→关系识别→｛关系列表｝

locs = [('Omnicom', 'IN', 'New York'),
        ('DDB Needham', 'IN', 'New York'),
        ('Kaplan Thaler Group', 'IN', 'New York'),
        ('BBDO South', 'IN', 'Atlanta'),
        ('Georgia-Pacific', 'IN', 'Atlanta')]
query = [e1 for (e1, rel, e2) in locs if e2 == 'Atlanta']
print(query)


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)  # 句子分割
    sentences = [nltk.word_tokenize(sent) for sent in sentences]  # 单词分割
    sentences = [nltk.pos_tag(sent) for sent in sentences]  # 词性标注


# 2 分块：用于实体识别的基本技术（P284 图7-2）
# * 小框显示词级标识符和词性标注
# * 大框表示组块（chunk），是较高级别的程序分块
# 分块构成的源文本中的片段不能重叠
# 正则表达式和N-gram方法分块；
# 使用CoNLL-2000分块语料库开发和评估分块器；

# 2.1 名词短语分块（NP-chunking，即“NP-分块”）寻找单独名词短语对应的块
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
# 定义分块语法
grammar = 'NP: {<DT>?<JJ>*<NN>}'
# 创建组块分析器
cp = nltk.RegexpParser(grammar)
# 使用分析器对句子进行分块
sentence = [('the', 'DT'), ('little', 'JJ'), ('yellow', 'JJ'), ('dog', 'NN'), ('barked', 'VBD'), ('at', 'IN'),
            ('the', 'DT'), ('cat', 'NN')]
result = cp.parse(sentence)
# 输出分块的树状图
print(result)
result.draw()

# 2.2. 标记模式
# 华尔街日报
sentence = [('another', 'DT'), ('sharp', 'JJ'), ('dive', 'NN'),
            ('trade', 'NN'), ('figures', 'NNS'),
            ('any', 'DT'), ('new', 'JJ'), ('policy', 'NN'), ('measures', 'NNS'),
            ('earlier', 'JJR'), ('stages', 'NNS'),
            ('Panamanian', 'JJ'), ('dictator', 'NN'), ('Manuel', 'NNP'), ('Noriega', 'NNP')]
grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)

# Grammar中输入语法，语法格式{<DT>?<JJ.*>*<NN.*>+}，不能在前面加NP:，具体可以参考右边的Regexps说明
# Development Set就是开发测试集，用于调试语法规则。绿色表示正确匹配，红色表示没有正确匹配。黄金标准标注为下划线
nltk.app.chunkparser()

# 2.3 用正则表达式分块（组块分析）Ex7-2 简单的名词短语分类器
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
            ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
# 两个规则组成的组块分析语法，注意规则执行会有先后顺序，两个规则如果有重叠部分，以先执行的为准
grammar = r'''
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
'''
grammar = r'NP: {<[CDJ].*>+}'
grammar = r'NP: {<[CDJN].*>+}'
grammar = r'NP: {<[CDJNP].*>+}'

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)

# 如果模式匹配位置重叠，最左边的优先匹配。
# 例如：如果将匹配两个连贯名字的文本的规则应用到包含3个连贯名词的文本中，则只有前两个名词被分块
nouns = [('money', 'NN'), ('market', 'NN'), ('fund', 'NN')]
grammar = 'NP: {<NN><NN>}'  # 错误分块的语法
grammar = 'NP: {<NN>+}'  # 正确分块的语法
cp = nltk.RegexpParser(grammar)
result = cp.parse(nouns)
print(result)

# 2.4 探索文本语料库：从已经标注的语料库中提取匹配特定词性标记序列的短语
cp = nltk.RegexpParser('CHUNK: {<V.*><TO><V.*>}')
brown = nltk.corpus.brown
for sent in brown.tagged_sents():
    tree = cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label() == 'CHUNK': print(subtree)


def find_chunks(pattern):
    cp = nltk.RegexpParser(pattern)
    brown = nltk.corpus.brown
    for sent in brown.tagged_sents():
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK' or subtree.label() == 'NOUNS':
                print(subtree)


find_chunks('CHUNK: {<V.*><TO><V.*>}')
find_chunks('NOUNS: {<N.*>{4,}}')


def find_chunks(pattern):
    cp = nltk.RegexpParser(pattern)
    brown = nltk.corpus.brown
    for sent in brown.tagged_sents():
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK' or subtree.label() == 'NOUNS':
                yield subtree


i = 0
for subtree in find_chunks('CHUNK: {<V.*><TO><V.*>}'):
    if i < 10:
        i = i + 1
        print(subtree)

# 2.5. 添加缝隙：寻找需要排除的成分
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
            ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]

# 先分块，再加缝隙，才能得出正确的结果
grammar = r'''
    NP: 
        {<.*>+}         # Chunk everything
        }<VBD|IN>+{     # Chink sequences of VBD and IN
'''
# 先加缝隙，再分块，就不能得出正确的结果，只会得到一个块，效果与没有使用缝隙是一样的
grammar = r'''
    NP: 
        }<VBD|IN>+{     # Chink sequences of VBD and IN
        {<.*>+}         # Chunk everything
'''

grammar = r'''
    NP: 
        {<.*>+}         # Chunk everything
'''

cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

# 分块的表示：标记与树状图
# 作为标注放分析之间的中间状态（Ref：Ch8），块结构可以使用标记或者树状图来表示
# 使用最为广泛的表示是IOB标记：I（Inside，内部）；O（Outside，外部）；B（Begin，开始）。

# 3. 开发和评估分块器

# 3.1. 读取IOB格式 和 CoNLL2000 语料库

# 不能在text里面加入“空格”和“置表符”用来控制文本的格式
text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''

# 绘制块结构的树状图表示
nltk.chunk.conllstr2tree(text, chunk_types = ('NP',)).draw()
nltk.chunk.conllstr2tree(text, chunk_types = ('NP', 'VP')).draw()

# CoNLL2000分块语料库包括3种分块类型：NP、VP、PP
from nltk.corpus import conll2000

train_sents = conll2000.chunked_sents('train.txt', chunk_types = 'NP')
train_sents = conll2000.chunked_sents('train.txt', chunk_types = ('NP', 'VP'))
train_sents = conll2000.chunked_sents('train.txt', chunk_types = ('NP', 'VP', 'PP'))
print(train_sents[0])

# 3.2. 简单的评估和基准
# 建立基准
test_sents = conll2000.chunked_sents('test.txt', chunk_types = 'NP')
print(test_sents[0])

cp = nltk.RegexpParser('')  # 没有任何语法规则，即所有的词都被标注为O
print(cp.evaluate(test_sents))

grammar = r'NP: {<[CDJNP].*>+}'
cp = nltk.RegexpParser(grammar)
test_result = cp.parse(test_sents)
print(test_result)
print(cp.evaluate(test_sents))


# P293 Ex7-4 unigram 标注器对名词短语分块
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]  # 准备训练用的数据
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)  # 使用训练数据训练一元语法标注器

    def parse(self, sentence):
        # print(sentence)
        pos_tags = [pos for (word, pos) in sentence]
        # print(pos_tags)       # 需要标注的内容 ['NN','CC','DT','PRP'...]
        tagged_pos_tag = self.tagger.tag(pos_tags)
        # print(tagged_pos_tag)   # 标注好的结果 [('NNP','I-NP'),(',','O')...]
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tag]  # 把标注好的结果选出来
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]  # 组成最后需要输出的结果
        # print(conlltags)          # 最后输出的结果：[('Rockwell', 'NNP', 'I-NP'), ('International', 'NNP', 'I-NP')...]
        return nltk.chunk.conlltags2tree(conlltags)  # 将结果转化成树块的方式输出


from nltk.corpus import conll2000

test_sents = conll2000.chunked_sents('test.txt', chunk_types = 'NP')
train_sents = conll2000.chunked_sents('train.txt', chunk_types = 'NP')
# 训练用的数据格式
train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
print(train_data[0])
# 评估unigram标注器的性能
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

# 直接对parse()函数进行测试
tmp_sents = conll2000.tagged_sents('test.txt')
print(tmp_sents[0])
unigram_chunker.parse(tmp_sents[0])

# 一元标注器对于标签的标注结果
postags = sorted(set(pos for sent in train_sents for (word, pos) in sent.leaves()))
print(unigram_chunker.tagger.tag(postags))


# 试着自己建立一个二元标注器
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tag = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tag]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


# 二元标注器对性能的提高非常有限
bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))


# 3.3 使用分类器的分块器的训练
# 想要最大限度地提升分块的性能，需要使用词的内容信息作为词性标记的补充。
# Ex 7.5. 使用连续分类器（最大熵分类器）对名词短语分块（i5-5200U，执行时间20分钟）
# 不能使用megam算法，megam表示LM-BFGS algorithm，需要使用External Libraries，
# Windows用户就不要尝试了，因为作者根本没有提供Windows的安装版本
# 取消algorithm='megam'设置，使用默认的算法就可以了-->The default algorithm = 'IIS'(Improved Iterative Scaling )
# ConsecutiveNPChunkTagger与Ex6-5中的ConsecutivePosTagger类相同，区别只有特征提取器不同。
class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                # print(untagged_sent, i, history)
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        # self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)
        self.classifier = nltk.MaxentClassifier.train(train_set, trace = 0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


# 对ConsecutiveNPChunkTagger的包装类，使之变成一个分块器
class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c,) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


# 1） 特征提取器，最为简单，只使用了单词本身的标签作为特征，训练结果与unigram分类器非常相似
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {'pos': pos}


from nltk.corpus import conll2000

test_sents = conll2000.chunked_sents('test.txt', chunk_types = 'NP')
train_sents = conll2000.chunked_sents('train.txt', chunk_types = 'NP')

# 最初的是[（（单词，标签），分块）,...]
chunked_sents = [[((w, t), c) for (w, t, c) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents[0:1]]
# 脱第一层“分块”得到[（单词，标签）,...]
tagged_sent = nltk.tag.untag(chunked_sents[0])
# 再脱一层“标签”得到[单词,...]
untagged_sent = nltk.tag.untag(tagged_sent)
# 再脱一层“标签”就只有报错了
nltk.tag.untag(untagged_sent)

history = []
for i, (word, tag) in enumerate(chunked_sents[0]):
    print(i, word, tag)
    featureset = npchunk_features(tagged_sent, i, history)
    print(featureset)
    history.append(tag)

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


# 2） 特征提取器，使用了单词前面一个单词的标签作为特征，效果类似于bigram分块器
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    return {'pos': pos, 'prevpos': prevpos}


chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


# 3） 特征提取器，使用了单词本身的标签、前一个单词、前一个单词的标签作为特征
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    return {'pos': pos, 'word': word, 'prevpos': prevpos}


chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


# 4) 特征提取器，使用了多种附加特征
#   * 预取特征
#   * 配对功能
#   * 复杂的语境特征
#   * tags-since-dt：用其创建一个字符串，描述自最近限定词以来遇到的所有词性标记
def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    if i == len(sentence) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sentence[i + 1]
    return {
            'pos': pos,
            'word': word,
            'prevpos': prevpos,
            'nextpos': nextpos,
            'prevpos+pos': '%s+%s' % (prevpos, pos),
            'pos+nextpos': '%s+%s' % (pos, nextpos),
            'tags-sincce-dt': tags_since_dt(sentence, i)
    }


chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

# 4. 语言结构中的递归
# 4.1 使用级联分块器构建嵌套的结构
# Ex7-6 分块器，处理NP（名词短语）PP（介绍短语）VP（动词短语）和$（句子的模式）
grammar = r'''
NP: {<DT|JJ|NN.*>+}             # Chunk sequences of DT, JJ, NN
PP: {<IN><NP>}                  # Chunk prepositions followed by NP
VP: {<VB.*><NP|PP|CLAUSE>+$}    # Chunk verbs and their arguments
CLAUSE: {<NP><VP>}              # Chunk NP, VP
'''

sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
            ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]

sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
            ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
            ("on", "IN"), ("the", "DT"), ("mat", "NN")]

cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

# 对于有深度的语法结构，一次剖析不能解决问题，使用loop设置循环剖析的次数
cp = nltk.RegexpParser(grammar, loop = 2)
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar, loop = 3)
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar, loop = 4)
parsed_sent = cp.parse(sentence)
parsed_sent.draw()
print(parsed_sent)

# 虽然级联过程可以创建深层结构，但是创建与调试过程非常困难，并且只能产生固定深度的树状图，仍然属于不完整的句法分析
# 因此，全面的剖析才更有效（Ref：Ch8）

# 4.2. 树状图：一组朴素连接的加标签的节点，从一个特殊的根节点沿一条唯一的路径到达每个节点。
# 创建树状图
tree1 = nltk.Tree('NP', ['Alice'])
print(tree1)

tree2 = nltk.Tree('NP', ['the', 'rabbit'])
print(tree2)

# 合并成大的树状图
tree3 = nltk.Tree('VP', ['chased', tree2])
tree4 = nltk.Tree('S', [tree1, tree3])

# 访问树状图的对象
print(tree4)
print(tree4[0])
print(tree4[1])
print(tree4[1][1])
print(tree4[1][1][1])

# 调用树状图的函数
print(tree4.label())
print(tree4.leaves())
print(tree4[1].label())
print(tree4[1].leaves())
print(tree4[1][1].label())
print(tree4[1][1].leaves())
print(tree4[1][1][1].label())
print(tree4[1][1][1].leaves())

tree4.draw()


# 4.3. 树的遍历：Ex7-7 递归函数遍历树状图
def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t, end = ' ')
    else:
        print('(', t.label(), end = ' ')
        for child in t:
            traverse(child)
        print(')', end = ' ')


# 不能使用Tree()函数直接基于字符串生成树了。
t = nltk.Tree.fromstring('(S (NP Alice) (VP chased (NP the rabbit)))')
t = nltk.Tree.fromstring(tree4.__str__())
print(t)
t.draw()
traverse(tree4)
traverse(t)

# 5. 命名实体识别：识别所有文本中提及的命名实体。
# 命名实体（Named Entity，NE）：是确切的名词短语，指特定类型的个体。
# 命名实体识别（Named Entity Recognition，NER）的两个子任务：
# 1）确定NE的边界
# 2）确定NE的类型
# NER的主要方法：查词典
# NER的主要困难：名称有歧义
# NER主要的技术手段：基于分类器进行分类

# NLTK提供的是已经训练好的可以识别命名实体的分类器
# 使用nltk.ne_chunk()函数调用分类器，binary=True表示标注为NE，否则会添加类型标签，例如：PERSON，GPE等等。
sent = nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary = True))
print(nltk.ne_chunk(sent))

# 6. 关系抽取：寻找指定类型的命名实体之间的关系
# 1）寻找所有（X，α，Y）形式的三元组，其中X和Y是指定类型的命名实体，α表示X和Y之间的关系的字符串
# 搜索包含词in的字符串
IN = re.compile(r'.*\bin')
# “(?!\b.+ing)”是一个否定预测先行断言，忽略如“success in supervising the transition of” 这样的字符串
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus = 'ieer', pattern = IN):
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
    for rel in nltk.sem.extract_rels('PER', 'ORG', doc, corpus = 'conll2002', pattern = VAN):
        # 抽取具备特定关系的命名实体
        clause = nltk.sem.clause(rel, relsym = 'VAN')
        # print(nltk.sem.clause(rel, relsym='VAN'))
        # 抽取具备特定关系的命名实体所在窗口的上下文
        rtuple = nltk.sem.rtuple(rel, lcon = True, rcon = True)
        # print(nltk.sem.rtuple(rel, lcon = True, rcon = True))

# 7 小结
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