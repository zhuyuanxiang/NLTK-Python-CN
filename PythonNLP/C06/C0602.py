# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0602.py
@Version    :   v0.1
@Time       :   2020-11-21 11:05
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

from tools import show_subtitle

# 6.2 有监督分类的应用场景
# 6.2.1 句子分割（标点符号的分类任务，遇到可能会结束句子的符号时，二元判断是否应该断句）

# 第一步：获得已经被分割成句子的数据

sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
# 标注所有句子结束符号的位置
for sent in sents:
    tokens.extend(sent)  # 句子标识符的合并链表，把所有原始的句子都合并成单词列表
    offset += len(sent)  #
    boundaries.add(offset - 1)  # 包含所有句子-边界标识符索引的集合

sorted(boundaries)[:10]


def punct_features(tokens, i):
    """标点的数据特征
    标点（punctuation）符号的特征
    :param tokens: 已经分词的标记集
    :type tokens:
    :param i: 需要抽取特征的标点符号的位置
    :type i:
    :return:
    :rtype:
    """
    return {
        'next-word-capitalized': tokens[i + 1][0].isupper(),
        'prevword': tokens[i - 1].lower(),
        'punct': tokens[i],
        'prev-word-is-one-char': len(tokens[i - 1]) == 1
    }


# 第二步：建立标点符号的特征集合
feature_sets = [(punct_features(tokens, i), (i in boundaries))
                for i in range(1, len(tokens) - 1)
                if tokens[i] in '.?!']

# 使用这些特征集，训练和评估一个标点符号分类器
size = int(len(feature_sets) * 0.1)
train_set, test_set = feature_sets[size:], feature_sets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# Ex6-6 基于分类的断句器
# ToDo: 原理是基于分类器对句子进行分类，但是没提供用于测试的数据
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in words:
        if word in '.?!' and classifier.classify((punct_features(words, i)) == True):
            sents.append(words[start:i + 1])
        if start < len(words):
            sents.append(words[start:])
            return sents


# 2.2. 识别对话行为类型
# 对话的行为类型
# Statement, System, Greet, Emotion, ynQuestion, whQuestion, Accept, Bye, Emphasis, Continuer, Reject, yAnswer, nAnswer, Clarify, Other
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
for i, post in enumerate(posts):
    if i < 10:
        print(i, ')', post)


def dialogue_act_features(post):
    """特征提取器，检查帖子包含哪些词"""
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


feature_sets = [
    (dialogue_act_features(post.text), post.get('class'))
    for post in posts
]

# 常用的对话行为分类
classes = [category for _, category in feature_sets]
classes_fd = nltk.FreqDist(classes)
classes_fd.most_common()

size = int(len(feature_sets) * 0.1)
train_set, test_set = feature_sets[size:], feature_sets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# 2.3. 识别文字蕴涵 (Recognizing textual entailment, RTE)
# 判断文本T内的一个给定片段是否继承着另一个叫做“假设”的文本
# 文字和假设之间的关系并不一定是逻辑蕴涵，而是人们是否会得出结论：文本提供的合理证据证明假设是真实的
# 可以把RTE当作一个分类任务，尝试为每一对预测“True”/“False”标签
# “True”表示保留了蕴涵；“False”表示没有保留蕴涵

# Ex6-7：“认识文字蕴涵”的特征提取器
def rte_features(rtepair):
    """
    词（即词类型）作为信息的代理，计数词重叠的程度和假设中有而文本没有的词的程度
    特征词包括（命名实体、）
    :param rtepair:
    :type rtepair:
    :return:
    :rtype:
    """
    # RTEFeatureExtractor类建立了一个词汇包
    # 这个词汇包在文本和假设中都有的，并且已经除去了一些停用词
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    # 计算 重叠性 和 差异性
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features


# 取出文本-假设对的数据
rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
extractor = nltk.RTEFeatureExtractor(rtepair)
show_subtitle("文本中的单词")
print(extractor.text_words)
show_subtitle("假设中的单词")
print(extractor.hyp_words)
show_subtitle("文本和假设中重叠的单词（非实体词）")
print(extractor.overlap('word'))
show_subtitle("文本和假设中重叠的实体词")
print(extractor.overlap('ne'))
show_subtitle("文本和假设中差异的单词（非实体词）")
print(extractor.hyp_extra('word'))
show_subtitle("文本和假设中差异的实体词")
print(extractor.hyp_extra('ne'))

# 2.4 扩展到大型的数据集
# NLTK提供对专业的机器学习软件包的支持，调用它们会比NLTK提供的分类器性能更好
