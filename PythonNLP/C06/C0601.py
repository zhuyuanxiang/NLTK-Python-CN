# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0601.py
@Version    :   v0.1
@Time       :   2020-11-20 16:30
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import random

import nltk
from nltk.corpus import names


# Chap6 学习分类文本
# 学习目标：
# 1.  识别出语言数据中可以用于分类的特征
# 2.  构建用于自动执行语言处理任务的语言模型
# 3.  从语言模型中学习与语言相关的知识

# 6.1 有监督分类
# 分类是为了给定的输入选择正确的类标签。
# 监督式分类：建立在训练语料基础之上的分类

# 6.1.1 性别鉴定
# 性别鉴定：以名字的最后一个字母为特征
def gender_features(word):
    return {'last_letter': word[-1]}


gender_features('Shrek')

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

# 特征数据集合
feature_sets = [
        (gender_features(n), gender)
        for (n, gender) in labeled_names
]

# 训练数据集合 和 测试数据集合
train_set, test_set = feature_sets[500:], feature_sets[:500]

# 朴素贝叶斯分类器训练和分类
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Neo is ", classifier.classify(gender_features('Neo')))
print("Trinity is ", classifier.classify(gender_features('Trinity')))

# 朴素贝叶斯分类器性能评估
print(nltk.classify.accuracy(classifier, test_set))

# 信息量大的特征
classifier.show_most_informative_features(5)


# 性别鉴定：以名字的长度为特征
def gender_features(word):
    return {'name length': len(word)}


gender_features('Shrek')


def gender_classifier():
    # 特征数据集合
    feature_sets = [
            (gender_features(n), gender)
            for (n, gender) in labeled_names
    ]

    # 训练数据集合 和 测试数据集合
    train_set, test_set = feature_sets[500:], feature_sets[:500]

    # 朴素贝叶斯分类器训练和分类
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print('Neo is', classifier.classify(gender_features('Neo')))
    print('Trinity is', classifier.classify(gender_features('Trinity')))

    # 朴素贝叶斯分类器性能评估
    print(nltk.classify.accuracy(classifier, test_set))

    # 信息量大的特征（发现特征信息量比较小，说明这个特征效果不好）
    classifier.show_most_informative_features(5)


gender_classifier()

# 在处理大型语料库时，构建包含所有实例特征的单独链表会占用大量的内存
# apply_features 返回一个链表，但是不会在内存中存储所有特征集的对象
from nltk.classify import apply_features

train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])
