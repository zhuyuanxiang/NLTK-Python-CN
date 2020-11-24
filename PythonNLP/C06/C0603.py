# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0603.py
@Version    :   v0.1
@Time       :   2020-11-23 12:04
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import random

import nltk
from nltk.corpus import brown
from nltk.corpus import names

from tools import show_subtitle


# Chap6 学习分类文本
# 学习目标：
# 1.  识别出语言数据中可以用于分类的特征
# 2.  构建用于自动执行语言处理任务的语言模型
# 3.  从语言模型中学习与语言相关的知识

# 6.3 评估 (Evaluation)
# 6.3.1 测试集
# 这种方式构建的数据集会导致训练集与测试集中的句子取自同一篇文章中，结果就是句子的风格相对一致
# 从而产生过拟合（即泛化准确率过高，与实际情况不符），影响分类器向其他数据集的推广
def split_all():
    show_subtitle("直接基于所有数据进行分割")
    tagged_sents = list(brown.tagged_sents(categories='news'))
    random.shuffle(tagged_sents)
    size = int(len(tagged_sents) * 0.1)
    train_set, test_set = tagged_sents[size:], tagged_sents[:size]
    return train_set, test_set


# 这种方式从文章层次将数据分开，就不会出现上面分解数据时出现的问题
def split_file_ids():
    show_subtitle("基于文章名称进行数据分割")
    file_ids = brown.fileids(categories='news')
    size = int(len(file_ids) * 0.1)
    train_set, test_set = brown.tagged_sents(file_ids[size:]), brown.tagged_sents(file_ids[:size])
    return train_set, test_set


# 直接从不同的类型中取数据，效果更好
def split_categories():
    show_subtitle("基于文章类型进行数据分割")
    train_set, test_set = brown.tagged_sents(categories='news'), brown.tagged_sents(categories='fiction')
    return train_set, test_set


# 6.3.2 准确度

# 下面这个是用于将List从二维转换成一维
# import operator
# from functools import reduce
#
# train_set, test_set = split_categories()
# b_tmp_set = reduce(operator.add, train_set)


def gender_features(word):
    return {'prefix1': word[0:1], 'prefix2': word[0:2], 'suffix1': word[-1:], 'suffix2': word[-2:]}


# 原始数据集合
labeled_names = (
        [
            (name, 'male')
            for name in names.words('male.txt')
        ]
        +
        [
            (name, 'female')
            for name in names.words('female.txt')
        ]
)
# 乱序排序数据集
random.shuffle(labeled_names)

feature_sets = [
    (gender_features(n), gender)
    for (n, gender) in labeled_names
]
train_set, test_set = feature_sets[500:], feature_sets[:500]
# 准确度：用于评估分类的质量，这里不能使用从 brown 提供的数据集，应该使用的第1节中数据集（名字：性别）来训练和测试
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('准确度: {:4.3f}'.format(nltk.classify.accuracy(classifier, test_set)))


# 6.3.3 精确度 和 召回率
# -   真阳性：是相关项目中正确识别为相关的（True Positive，TP）
# -   真阴性：是不相关项目中正确识别为不相关的（True Negative，TN）
# -   假阳性：（I型错误）是不相关项目中错误识别为相关的（False Positive，FP）
# -   假阴性：（II型错误）是相关项目中错误识别为不相关的（False Negative，FN）
# -   精确度：（Precision）表示发现的项目中有多少是相关的，TP/（TP+FP）
# -   召回率：（Recall）表示相关的项目中发现了多少，TP/（TP+FN）
# -   F-度量值（F-Measure）：也叫F-得分（F-Score），组合精确度和召回率为一个单独的得分
#       被定义为精确度和召回率的调和平均数(2*Precision*Recall)/(Precision+Recall)=2*/(1/Precision + 1/Recall)

# 6.3.4 混淆矩阵
def tag_list(tagged_sents):
    return [
        tag
        for sent in tagged_sents
        for (word, tag) in sent
    ]


def apply_tagger(tagger, corpus):
    return [
        # 删除已经标注的标签，方便测试
        tagger.tag(nltk.tag.untag(sent))
        # nltk.tag.untag() 只有对句子进行去标签处理
        for sent in corpus
    ]


train_sents = brown.tagged_sents(categories='editorial', tagset='universal')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
gold = tag_list(train_sents)
test = tag_list(apply_tagger(t2, train_sents))
confusion_matrix = nltk.ConfusionMatrix(gold, test)
print(confusion_matrix)

# 6.3.4 交叉验证
# 将原始语料细分为N个子集，在不同的测试集上执行多重评估，然后组合这些评估的得分。
# 交叉验证的作用：
# 1） 解决数据集合过小的问题，
# 2） 研究不同的训练集上性能变化有多大

# 接下来的三节中，将研究三种机器学习分类模型：决策树、朴素贝叶斯分类器 和 最大熵分类器
# 仔细研究这些分类器的收获：
# 1）如何基于一个训练集上的数据来选择合适的学习模型
# 2）如何选择合适的特征应用到相关的学习模型上
# 3）如何提取和编码特征，使之包含最大的信息量，以及保持这些特征之间的相关性
