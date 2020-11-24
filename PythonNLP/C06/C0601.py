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

from tools import show_subtitle

# Chap6 学习分类文本
# 学习目标：
# 1.  识别出语言数据中可以用于分类的特征
# 2.  构建用于自动执行语言处理任务的语言模型
# 3.  从语言模型中学习与语言相关的知识

# 6.1 有监督分类
# 分类是为了给定的输入选择正确的类标签。
# 监督式分类：建立在训练语料基础之上的分类

# 6.1.1 性别鉴定
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


# 信息量大的特征
# 'SklearnClassifier' object has no attribute 'show_most_informative_features'
# classifier.show_most_informative_features(5)

def gender_classifier(features):
    # 特征数据集合
    feature_sets = [
        (features(n), gender)
        for (n, gender) in labeled_names
    ]

    # 训练数据集合 和 测试数据集合
    train_set, test_set = feature_sets[500:], feature_sets[:500]

    # 使用 Scikit-Learn 的 GaussNB 朴素贝叶斯 分类器进行分类
    from sklearn.naive_bayes import GaussianNB
    from nltk.classify.scikitlearn import SklearnClassifier

    classifier = SklearnClassifier(GaussianNB(), sparse=False).train(train_set)
    print("Scikit-GaussNB 分类器性能评估= ", nltk.classify.accuracy(classifier, test_set))

    # 使用 NLTK 的 朴素贝叶斯 分类器进行分类训练
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("NLTK-NB 分类器性能评估= ", nltk.classify.accuracy(classifier, test_set))

    # 信息量大的特征（发现特征信息量比较小，说明这个特征效果不好）
    show_subtitle("信息量大的特征")
    classifier.show_most_informative_features(5)

    show_subtitle("Neo's feature")
    print(features('Neo'))
    print("Neo is", classifier.classify(features('Neo')))
    show_subtitle("Trinity's feature")
    print(features('Trinity'))
    print("Trinity is", classifier.classify(features('Trinity')))


# 以名字的最后一个字母为特征
gender_classifier(lambda name: {'last_letter': name[-1]})

# 性别鉴定：以名字的长度为特征
gender_classifier(lambda name: {'name length': len(name)})


# 在处理大型语料库时，构建包含所有实例特征的单独链表会占用大量的内存
# apply_features 返回一个链表，但是不会在内存中存储所有特征集的对象
def gender_features(name):
    return {'last_letter': name[-1]}


from nltk.classify import apply_features

train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])

# 6.1.2 选择正确的特征

# 两个特征可以少量地增加了准确率（0.776>0.774)
gender_classifier(lambda name: {'first_letter': name[0].lower(), 'last_letter': name[-1].lower()})


# 复杂的特征集，导致过拟合，反而降低了准确率
def gender_features(name):
    # 特征集合
    features = {'first_letter': name[0].lower(), 'last_letter': name[-1].lower()}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count({})'.format(letter)] = name.lower().count(letter)
        features['has({})'.format(letter)] = (letter in name.lower())
    return features


gender_classifier(gender_features)


# 开发测试集的作用
# 训练集：训练模型；
# 开发测试集：执行错误分析；
# 测试集：系统的最终评估。
def train_dev_test():
    train_names = labeled_names[1500:]
    devtest_names = labeled_names[500:1500]
    test_names = labeled_names[:500]

    train_set = [
        (gender_features(n), gender)
        for (n, gender) in train_names
    ]
    devtest_set = [
        (gender_features(n), gender)
        for (n, gender) in devtest_names
    ]
    test_set = [
        (gender_features(n), gender)
        for (n, gender) in test_names
    ]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, devtest_set))
    print(nltk.classify.accuracy(classifier, test_set))

    # 使用开发测试集执行错误分析，避免使用测试集进行错误分析和修正是为了避免对测试集的过拟合
    # 因为利用测试集的分析来增加规则，修正错误，则会使测试集的评估数据自然变好，但是测试集不能代表所有的数据特征
    # 因此利用训练集训练模型，使用开发测试集进行错误分析和规则修正，再利用测试集来评估，就可以尽可能保证数据的独立性和分布的均衡性
    errors = []
    for (name, tag) in devtest_names:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append((tag, guess, name))

    for (tag, guess, name) in sorted(errors):
        print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))


# 性别鉴定的两个特征：
# 最后一个字母，最后两个字母，准确率更高（0.792）
gender_classifier(lambda name: {
    'suffix1': name[-1:],
    'suffix2': name[-2:]
})

# 性别鉴定的三个特征：
# 头两个字母，最后一个字母，最后两个字母，准确率更高（0.806）
gender_classifier(lambda name: {
    'prefix': name[0:2],
    'suffix1': name[-1:],
    'suffix2': name[-2:]
})

# 性别鉴定的四个特征：
# 头一个字母，头两个字母，最后一个字母，最后两个字母，准确率更高（0.814）
gender_classifier(lambda name: {
    'prefix1': name[0:1],
    'prefix2': name[0:2],
    'suffix1': name[-1:],
    'suffix2': name[-2:]
})

# 6.1.3 文档分类
# 学习语料库中的标记，为新文档分配类别标签

# 使用电影评论语料库，将每个评论归类为正面或者负面
from nltk.corpus import movie_reviews

show_subtitle("categories()")
print(movie_reviews.categories())
show_subtitle("fileids('neg')")
print(movie_reviews.fileids('neg'))
show_subtitle("words('/neg/cv995_23113.txt'")
print(movie_reviews.words('neg/cv995_23113.txt'))

documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

# Ex6-2 文档分类的特征提取器，其特征表示每个词是否在一个给定的文档中
# 评论中使用的所有单词
all_words = nltk.FreqDist(
    w.lower()
    for w in movie_reviews.words()
)
show_subtitle("all_words.most_common(20)")
print(all_words.most_common(20))
# 取出部分单词作为特征
# 任取2000个单词就可以产生很好的结果：0.81
word_features_random = list(all_words)[:2000]
show_subtitle("word_features[:20]")
print(word_features_random[:20])


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


myDocument = movie_reviews.words('pos/cv957_8737.txt')
myFeature = document_features(myDocument, word_features_random)
for i, (key, value) in enumerate(myFeature.items()):
    if i < 10:
        print(key, value)


# Ex6-3 训练和测试分类器以进行文档分类
def document_classifier(word_features):
    feature_sets = [
        (document_features(d, word_features), c)
        for (d, c) in documents
    ]
    train_set, test_set = feature_sets[100:], feature_sets[:100]

    # 使用 Scikit-Learn 的 GaussNB 朴素贝叶斯 分类器进行分类
    from sklearn.naive_bayes import GaussianNB
    from nltk.classify.scikitlearn import SklearnClassifier

    classifier = SklearnClassifier(GaussianNB(), sparse=False).train(train_set)
    print("Scikit-GaussNB 分类器性能评估= ", nltk.classify.accuracy(classifier, test_set))

    # 使用 NLTK 的 朴素贝叶斯 分类器进行分类训练
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("NLTK-NB 分类器性能评估= ", nltk.classify.accuracy(classifier, test_set))

    # 信息量大的特征（发现特征信息量比较小，说明这个特征效果不好）
    show_subtitle("信息量大的特征")
    classifier.show_most_informative_features(5)


document_classifier(word_features_random)

# 6.1.4 词性标注，词类标注，(Part-of-Speech Tagging, POST)
from nltk.corpus import brown

# 寻找最常见的单词后缀，可能也是最有信息量的单词后缀
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
common_suffixes = [
    suffix
    for (suffix, count) in suffix_fdist.most_common(100)
]
show_subtitle("common_suffixes[:20]")
print(common_suffixes[:20])


# 定义特征提取函数，以前100个单词后缀为特征，确定每个单词受这100个后缀的影响
# 例如：should的后缀'ld'存在，后缀'en'不存在
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features


# 建立特征集合，每个单词的特征（后缀'en':存在）+标注（NN)为一条数据，组成的特征集合
tagged_words = brown.tagged_words(categories='news')
feature_sets = [(pos_features(n), g) for (n, g) in tagged_words]

size = int(len(feature_sets) * 0.1)
train_set, test_set = feature_sets[size:], feature_sets[:size]
# 下面这行代码执行时间过长，可以尝试减少 word_features 中的特征数量
# i5-5200U 需要运行10分钟以上！！！
# 决策树模型的优点：容易解释，甚至可以使用伪代码的形式输出
classifier = nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.classify(pos_features('cats'))
print(classifier.pseudocode(depth=4))

# 使用 Scikit-Learn 运行速度会快很多
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import SklearnClassifier

classifier = SklearnClassifier(DecisionTreeClassifier()).train(train_set)
print("Scikit 决策树分类器性能评估= ", nltk.classify.accuracy(classifier, test_set))
print("cats 标注的结果：", classifier.classify(pos_features('cats')))


# 6.1.5 探索上下文语境
# -   通过上下文提高分类的精度，例如：large后面的可能是名词；
# -   但是不能根据前词的标记判断下个词的类别，例如：前面是形容词，后面的可能是名词
# -   下节的序列分类就是使用联合分类器模型，为一些相关的输入选择适当的标签

# Ex6-4 词性分类器，特征检测器通过一个单词的上下文来决定这个单词的词性标记
def pos_features(sentence: list, i: int) -> dict:
    """更加复杂的特征提取函数
    :param sentence:用于提取单词的句子，提供句子才可以提供上下文语境特征
    :type sentence:list
    :param i:句子中需要提取特征的单词的位置
    :type i:int
    :return:相应单词的特征集
    :rtype:dict """
    # 当前词的后缀特征
    features = {
        'Suffix(1)': sentence[i][-1:],
        'Suffix(2)': sentence[i][-2:],
        'Suffix(3)': sentence[i][-3:]
    }
    # 单词前面的词作为特征，如果单词为头一个单词就设置前面一个单词为<START>
    if i == 0:
        features['prev-word'] = '<START>'
    else:
        features['prev-word'] = sentence[i - 1]
    return features


# (brown.sents()[0],8) == 'investigation'
type(pos_features(brown.sents()[0], 8))
tagged_sents = brown.tagged_sents(categories='news')
feature_sets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        feature_sets.append((pos_features(untagged_sent, i), tag))  # 特征集合中的元素必须是tuple的形式

size = int(len(feature_sets) * 0.1)
train_set, test_set = feature_sets[size:], feature_sets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# 6.1.6 序列分类（通过上下文的标签提高分类的精度）
# 为了获取相关分类任务之间的依赖关系，可以使用联合分类器模型
# 连续分类或者贪婪序列分类的序列分类器策略
# 得到更高的分类准确率

# Ex6-5 使用连贯分类器进行词性标注
def pos_features(sentence: list, i: int, history: list) -> dict:
    """
        >>> pos_features(['I', 'love', 'you','.'], 1, ['N'])
        {'Suffix(1)': 'e', 'Suffix(2)': 've', 'Suffix(3)': 'ove', 'prev-word': 'I', 'prev-tag': 'N'}
    :param sentence:用于提取单词的句子，提供句子才可以提供上下文语境特征
    :type sentence:list
    :param i:句子中需要提取特征的单词的位置
    :type i:int
    :param history:history中的每个标记对应sentence中的一个词，但是history只包括已经归类的词的标记
    :type history:
    :return:相应单词的特征集
    :rtype:dict
    """
    features = {
        'Suffix(1)': sentence[i][-1:],
        'Suffix(2)': sentence[i][-2:],
        'Suffix(3)': sentence[i][-3:]
    }
    if i == 0:
        features['prev-word'] = '<START>'
        features['prev-tag'] = '<START>'
    else:
        features['prev-word'] = sentence[i - 1]
        features['prev-tag'] = history[i - 1]
    return features


history = ['N']
pos_features(brown.sents()[0], 1, history)


class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            # print(untagged_sent)  # 将（单词，标签）元组组成的列表中的标签去掉，只保留单词组成的列表
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featuresets = pos_features(untagged_sent, i, history)
                train_set.append((featuresets, tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featuresets = pos_features(sentence, i, history)
            tag = self.classifier.classify(featuresets)
            history.append(tag)
        return zip(sentence, history)


tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
print("train_sents[0]= ", train_sents[0])

nltk.tag.untag(train_sents[0])
for i, (word, tag) in enumerate(train_sents[0]):
    if i < 10:
        print(i, word, tag)

tagger = ConsecutivePosTagger(train_sents)
print(tagger.evaluate(test_sents))

### 6.1.7 序列分类器中的其他方法
# 1.  基于转换的联合分类：Ref：Sec5.6节描述的Brill标注器，解决了前面分类器一旦分类就无法改变的问题
# 2.  基于HMM的联合分类：
#       -   特点：不仅考虑了单词的上下文，还可以考虑更长的依赖性
#       -   输出：不是某个单词标记的最大可能性，而是整个序列中所有单词标记的最大可能性
#       -   常用的模型：最大熵马尔可夫模型 和 线性链条件随机场模型
