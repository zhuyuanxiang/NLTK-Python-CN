# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0703.py
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

# 7.3 开发和评估分块器

# 7.3.1 读取 IOB格式 和 CoNLL2000 语料库

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
nltk.chunk.conllstr2tree(text, chunk_types=('NP',)).draw()
nltk.chunk.conllstr2tree(text, chunk_types=('NP', 'VP')).draw()

# CoNLL2000分块语料库包括3种分块类型：NP、VP、PP
from nltk.corpus import conll2000

train_sents = conll2000.chunked_sents('train.txt', chunk_types='NP')
print(train_sents[0])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=('NP', 'VP'))
print(train_sents[0])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=('NP', 'VP', 'PP'))
print(train_sents[0])

# 7.3.2 简单的评估和基准
# 建立基准
test_sents = conll2000.chunked_sents('test.txt', chunk_types='NP')
print(test_sents[0])

# 没有任何语法规则，即所有的词都被标注为O
print(nltk.RegexpParser('').evaluate(test_sents))

# 正则表达式分块器
grammar = r'NP: {<[CDJNP].*>+}'
print(nltk.RegexpParser(grammar).parse(test_sents))
print(nltk.RegexpParser(grammar).evaluate(test_sents))


# P293 Ex7-4 unigram 标注器对名词短语分块
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [
            [(t, c)
             for w, t, c in nltk.chunk.tree2conlltags(sent)]  # 准备训练用的数据
            for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)  # 使用训练数据训练一元语法标注器

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        # 需要标注的内容 ['NN','CC','DT','PRP'...]
        tagged_pos_tag = self.tagger.tag(pos_tags)
        # 标注好的结果 [('NNP','I-NP'),(',','O')...]
        chunktags = [
            chunktag
            for (pos, chunktag) in tagged_pos_tag
        ]  # 把标注好的结果选出来
        conlltags = [
            (word, pos, chunktag)
            for ((word, pos), chunktag) in zip(sentence, chunktags)
        ]  # 组成最后需要输出的结果
        # 最后输出的结果：[('Rockwell', 'NNP', 'I-NP'), ('International', 'NNP', 'I-NP')...]
        return nltk.chunk.conlltags2tree(conlltags)  # 将结果转化成树块的方式输出


from nltk.corpus import conll2000

test_sents = conll2000.chunked_sents('test.txt', chunk_types='NP')
train_sents = conll2000.chunked_sents('train.txt', chunk_types='NP')

# 评估unigram标注器的性能
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

# 训练用的数据格式
train_data = [
    [(t, c)
     for w, t, c in nltk.chunk.tree2conlltags(sent)]
    for sent in train_sents]
print(train_data[0])

# 测试类 UnigramChunker 的 parse() 函数
# ToDo：Error
tmp_sents = conll2000.tagged_sents('test.txt')
print(tmp_sents[0])
unigram_chunker.parse(tmp_sents[0])

# 一元标注器对于标签的标注结果
postags = sorted(set(
    pos
    for sent in train_sents
    for (word, pos) in sent.leaves()))
print(unigram_chunker.tagger.tag(postags))


# 试着自己建立一个二元标注器
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [
            [(t, c)
             for w, t, c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [
            pos
            for (word, pos) in sentence]
        tagged_pos_tag = self.tagger.tag(pos_tags)
        chunktags = [
            chunktag
            for (pos, chunktag) in tagged_pos_tag]
        conlltags = [
            (word, pos, chunktag)
            for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


# 二元标注器对性能的提高非常有限
bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))


# 7.3.3 训练基于分类器的分块器
# 想要最大限度地提升分块的性能，需要使用词的内容信息作为词性标记的补充。

# Ex-7.5. 使用连续分类器（最大熵分类器）对名词短语分块（i5-5200U，执行时间20分钟）
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
        self.classifier = nltk.MaxentClassifier.train(train_set, trace=0)

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
        tagged_sents = [
            [((w, t), c) for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [
            (w, t, c,)
            for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


# 1） 第一个特征提取器
#       最为简单，只使用了单词本身的标签作为特征，训练结果与unigram分类器非常相似
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {'pos': pos}


from nltk.corpus import conll2000

test_sents = conll2000.chunked_sents('test.txt', chunk_types='NP')
train_sents = conll2000.chunked_sents('train.txt', chunk_types='NP')

# 验证基于分类器的分块器的性能
chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

# 最初的是[（（单词，标签），分块）,...]
chunked_sents = [
    [((w, t), c)
     for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
    for sent in train_sents[0:1]]
print(chunked_sents[0])

# 脱第一层“分块”得到[（单词，标签）,...]
tagged_sent = nltk.tag.untag(chunked_sents[0])
print(tagged_sent)

# 再脱一层“标签”得到[单词,...]
untagged_sent = nltk.tag.untag(tagged_sent)
print(untagged_sent)

# 再脱一层“标签”就只有报错了
# nltk.tag.untag(untagged_sent)

history = []
for i, (word, tag) in enumerate(chunked_sents[0]):
    print(i, word, tag)
    feature_set = npchunk_features(tagged_sent, i, history)
    print(feature_set)
    history.append(tag)


# 2） 第二个特征提取器
#       使用了单词前面一个单词的标签作为特征，效果类似于bigram分块器
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    return {'pos': pos, 'prevpos': prevpos}


chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


# 3） 第三个特征提取器，使用了单词本身的标签、前一个单词、前一个单词的标签作为特征
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i - 1]
    return {'pos': pos, 'word': word, 'prevpos': prevpos}


chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


# 4) 第四个特征提取器，使用了多种附加特征
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


