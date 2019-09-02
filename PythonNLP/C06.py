# Chap6 学习分类文本
# 学习目标：
# 1） 识别出语言数据中可以用于分类的特征
# 2） 构建用于自动执行语言处理任务的语言模型
# 3） 从语言模型中学习关于语言的知识

import random

import nltk
from nltk.corpus import brown
from nltk.corpus import names


# 1. 有监督分类
# 分类是为了给定的输入选择正确的类标签。
# 监督式分类：建立在训练语料基础之上的分类

# 性别鉴定：以名字的最后一个字母为特征
def gender_features(word):
    return {'last_letter': word[-1]}


gender_features('Shrek')

# 原始数据集合
labeled_names = ([(name, 'male') for name in names.words('male.txt')]
                 + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

# 特征数据集合
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# 训练数据集合 和 测试数据集合
train_set, test_set = featuresets[500:], featuresets[:500]

# 朴素贝叶斯分类器训练和分类
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

# 朴素贝叶斯分类器性能评估
print(nltk.classify.accuracy(classifier, test_set))

# 信息量大的特征
classifier.show_most_informative_features(5)


# 性别鉴定：以名字的长度为特征
def gender_features(word):
    return {'name length': len(word)}


def gender_classifier():
    # 特征数据集合
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

    # 训练数据集合 和 测试数据集合
    train_set, test_set = featuresets[500:], featuresets[:500]

    # 朴素贝叶斯分类器训练和分类
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print('Neo: {}'.format(classifier.classify(gender_features('Neo'))))
    print('Trinity: {}'.format(classifier.classify(gender_features('Trinity'))))

    # 朴素贝叶斯分类器性能评估
    print(nltk.classify.accuracy(classifier, test_set))

    # 信息量大的特征（发现特征信息量比较小，说明这个特征效果不好）
    classifier.show_most_informative_features(5)


gender_classifier()

from nltk.classify import apply_features

train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])
len(train_set)
len(test_set)


# 1.2. 选择正确的特征
# 两个特征可以少量地增加了准确率（0.776>0.774)
def gender_features(name):
    features = {}
    features['first_letter'] = name[0].lower()
    features['last_letter'] = name[-1].lower()
    return features


gender_classifier()


# 复杂的特征集，导致过拟合，反而降低了准确率
def gender_features(name):
    # 特征集合
    features = {}
    features['first_letter'] = name[0].lower()
    features['last_letter'] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count({})'.format(letter)] = name.lower().count(letter)
        features['has({})'.format(letter)] = (letter in name.lower())
    return features


gender_features('John')
gender_classifier()

# 开发测试集的作用
# 训练集：训练模型；
# 开发测试集：执行错误分析；
# 测试集：系统的最终评估。
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]
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
print(errors)

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))


# 性别鉴定的两个特征：最后一个字母，最后两个字母，准确率更高（0.792）
def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}


gender_classifier()


# 性别鉴定的三个特征：头两个字母，最后一个字母，最后两个字母，准确率更高（0.81）
def gender_features(word):
    return {'prefix': word[0:2], 'suffix1': word[-1:], 'suffix2': word[-2:]}


gender_classifier()


def gender_features(word):
    return {'prefix1': word[0:1], 'prefix2': word[0:2], 'suffix1': word[-1:], 'suffix2': word[-2:]}


gender_classifier()

# 1.3. 文档分类：学习语料库中的标记，为新文档分配类别标签
# 使用电影评论语料库，将每个评论归类为正面或者负面
from nltk.corpus import movie_reviews

movie_reviews.categories()
movie_reviews.fileids('neg')
movie_reviews.words('neg/cv995_23113.txt')

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Ex6-2 文档分类的特征提取器，其特征表示每个词是否在一个给定的文档中
# 评论中使用的所有单词
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
all_words.most_common(20)
# 取出部分单词作为特征
# 任取2000个单词就可以产生很好的结果：0.81
word_features = list(all_words)[:2000]
# 取出2000个高频单词产生更好的结果：0.82
word_features = [word for word, _ in all_words.most_common(2000)]
# 取出2000个低频单词产生更差的结果：0.69
word_features = [word for word, _ in all_words.most_common(20000)[-2000:]]


# most_common() 返回的不是单词集合，不能作为正确的特征使用，需要将特征做进一步处理
# word_features=list(all_words.most_common(2000))


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


myDocument = movie_reviews.words('pos/cv957_8737.txt')
myFeature = document_features(myDocument)
print(myFeature)

# Ex6-3 训练和测试分类器以进行文档分类
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

# 1.4. 词性标注，词类标注，(Part-of-Speech Tagging, POST)
# 寻找最常见的单词后缀，可能也是最有信息量的单词后缀
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]


# 定义特征提取函数，以前100个单词后缀为特征，确定每个单词受这100个后缀的影响
# 例如：should的后缀'ld'存在，后缀'en'不存在
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features


# 建立特征集合，每个单词的特征（后缀'en':存在）+标注（NN)为一条数据，组成的特征集合
tagged_words = brown.tagged_words(categories = 'news')
featuresets = [(pos_features(n), g) for (n, g) in tagged_words]

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
# 下面这行代码执行时间过长，可以尝试减少 word_features 中的特征数量
# i5-5200U 需要运行10分钟以上！！！
# 决策树模型的优点：容易解释，甚至可以使用伪代码的形式输出
classifier = nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.classify(pos_features('cats'))
print(classifier.pseudocode(depth = 4))


# 1.5. 探索上下文语境
# 通过上下文提高分类的精度，例如：large后面的可能是名词；
# 但是不能根据前词的标记判断下个词的类别，例如：前面是形容词，后面的可能是名词
# 下节的序列分类就是使用联合分类器模型，为一些相关的输入选择适当的标签

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
tagged_sents = brown.tagged_sents(categories = 'news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag))  # 特征集合中的元素必须是tuple的形式

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# 1.6. 序列分类（通过上下文的标签提高分类的精度）
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


tagged_sents = brown.tagged_sents(categories = 'news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]

train_sents[0]
nltk.tag.untag(train_sents[0])
for i,(word,tag) in enumerate(train_sents[0]):
    if i<10:
        print(i,word,tag)

tagger = ConsecutivePosTagger(train_sents)
print(tagger.evaluate(test_sents))

# 序列分类器中的其他方法
# 1） 基于转换的联合分类：Ref：Sec5.6节描述的Brill标注器，解决了前面分类器一旦分类就无法改变的问题
# 2） 基于HMM的联合分类：不仅考虑了单词的上下文，还可以考虑更长的依赖性
#       不是某个单词标记的最大可能性，而是整个序列中所有单词标记的最大可能性
#       比较常用的模型：最大熵马尔可夫模型 和 线性链条件随机场模型


# 2. 有监督分类的应用场景
# 2.1. 句子分割（标点符号的分类任务，遇到可能会结束句子的符号时，二元判断是否应该断句）
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
    """
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


# 建立标点符号的特征集合
featuresets = [(punct_features(tokens, i), (i in boundaries))
               for i in range(1, len(tokens) - 1)
               if tokens[i] in '.?!']

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# Ex6-6 基于分类的断句器（怎么用？）
# 原理是基于分类器对句子进行分类，但是没胡提供用于测试的数据
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in words:
        if word in '.?!' and classifier.classify((punct_features(words, i)) == True):
            sents.append(words[start:i + 1])
        if start < len(words):
            sents.append(words[start:])
            return sents


# segment_sentences((0,['I','love','u','.','You','love','me','.']))

# 2.2. 识别对话行为类型
# 对话的行为类型
# Statement, System, Greet, Emotion, ynQuestion, whQuestion, Accept, Bye, Emphasis, Continuer, Reject, yAnswer, nAnswer, Clarify, Other
posts = nltk.corpus.nps_chat.xml_posts()[:10000]


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
# 常用的对话行为分类
classes = [category for _, category in featuresets]
classes_fd = nltk.FreqDist(classes)
classes_fd.most_common()

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# 2.3. 识别文字蕴涵 (Recognizing textual entailment, RTE)
# 判断文本T内的一个给定片段是否继承着另一个叫做“假设”的文本
# 文字和假设之间的关系并不一定是逻辑蕴涵，而是人们是否会得出结论：文本提供的合理证据证明假设是真实的
# 可以把RTE当作一个分类任务，尝试为每一对预测“True”/“False”标签
# “True”表示保留了蕴涵；“False”表示没有保留蕴涵
def rte_features(rtepair):
    """
    词（即词类型）作为信息的代理，计数词重叠的程度和假设中有而文本没有的词的程度
    特征词包括（命名实体、）
    :param rtepair:
    :type rtepair:
    :return:
    :rtype:
    """
    # RTEFeatureExtractor类建立了一个在文本和假设中都有的并且已经除去了一些停用词后的词汇包
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
print(extractor.text_words)  # 文本中的单词
print(extractor.hyp_words)  # 假设中的单词
print(extractor.overlap('word'))  # 文本和假设中重叠的单词（非实体词）
print(extractor.overlap('ne'))  # 文本和假设中重叠的实体词
print(extractor.hyp_extra('word'))  # 文本和假设中差异的单词（非实体词）
print(extractor.hyp_extra('ne'))  # 文本和假设中差异的实体词

# 2.4 扩展到大型的数据集
# NLTK提供对专业的机器学习软件包的支持，调用它们会比NLTK提供的分类器性能更好

# 3. 评估 (Evaluation)
# 3.1. 测试集
# 这种方式构建的数据集会导致训练集与测试集中的句子取自同一篇文章中，结果就是句子的风格相对一致
# 从而产生过拟合（即泛化准确率过高，与实际情况不符），影响分类器向其他数据集的推广
tagged_sents = list(brown.tagged_sents(categories = 'news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]

# 这种方式从文章层次将数据分开，就不会出现上面分解数据时出现的问题
file_ids = brown.fileids(categories = 'news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])

# 直接从不同的类型中取数据，效果更好
train_set = brown.tagged_sents(categories = 'news')
test_set = brown.tagged_sents(categories = 'fiction')

# 下面这个是用于将List从二维转换成一维
import operator
from functools import reduce

b_tmp_set = reduce(operator.add, train_set)
classifier = nltk.NaiveBayesClassifier.train(b_tmp_set)


def gender_features(word):
    return {'prefix1': word[0:1], 'prefix2': word[0:2], 'suffix1': word[-1:], 'suffix2': word[-2:]}


featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
# 准确度：用于评估分类的质量，这里不能使用从 brown 提供的数据集，应该使用的第1节中数据集（名字：性别）来训练和测试
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set)))


# 3.3 精度度 和 召回率
# * 真阳性是相关项目中正确识别为相关的（True Positive，TP）
# * 真阴性是不相关项目中正确识别为不相关的（True Negative，TN）
# * 假阳性（I型错误）是不相关项目中错误识别为相关的（False Positive，FP）
# * 假阴性（II型错误）是相关项目中错误识别为不相关的（False Negative，FN）
# * 精确度（Precision）：表示发现的项目中有多少是相关的，TP/（TP+FP）
# * 召回率（Recall）：表示相关的项目中发现了多少，TP/（TP+FN）
# * F-度量值（F-Measure）：也叫F-得分（F-Score），组合精确度和召回率为一个单独的得分
#       被定义为精确度和召回率的调和平均数(2*Precision*Recall)/(Precision+Recall)

# 3.4 混淆矩阵
def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]


def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]


train_sents = brown.tagged_sents(categories = 'editorial', tagset = 'universal')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff = t0)
t2 = nltk.BigramTagger(train_sents, backoff = t1)
gold = tag_list(train_sents)
test = tag_list(apply_tagger(t2, train_sents))
cm = nltk.ConfusionMatrix(gold, test)
print(cm)

# 交叉验证：将原始语料细分为N个子集，在不同的测试集上执行多重评估，然后组合这些评估的得分。
# 交叉验证的作用：
# 1） 解决数据集合过小的问题，
# 2） 研究不同的训练集上性能变化有多大

# 接下来的三节中，将研究三种机器学习分类模型：决策树、朴素贝叶斯分类器 和 最大熵分类器
# 仔细研究这些分类器的收获：
# 1）如何基于一个训练集上的数据来选择合适的学习模型
# 2）如何选择合适的特征应用到相关的学习模型上
# 3）如何提取和编码特征，使之包含最大的信息量，以及保持这些特征之间的相关性

# 4 决策树：为输入值选择标签的简单流程图。流程图是由检查特征值的决策节点和分配标签的叶节点组成。（P263 图6-4）
# 为语料库选择最好的“决策树桩”。
# 决策树桩是只有一个节点的决策树，基于单个特征来决定如何为输入分类。
# 每个可能的特征值包含一个叶子，并为特征输入指定的类标签。
# 基于决策树桩来生成决策树算法模型：
# 1）选择分类任务的整体最佳的决策树桩
# 2）在训练集上检查每个叶子的准确度。
#       没有达到足够准确度的叶子会被新的决策树桩替换，新的决策树桩是在训练语料的子集上训练的，这些训练语库都是根据到叶子的路径来选择的。

# 4.1 熵和信息增益
# 使用信息增益来确定决策树桩最有信息量的特征。
# 信息增益就是原来的熵送去新减少的熵。信息增益越高，将输入值分为相关组的决策树桩，通过选择具有最高信息增益的决策树桩来建立决策树
# 决策树模型的优点：简单明了，容易理解。
# 决策树模型的缺点：基于分支划分数据过多后，导致数据量不足，产生过拟合；强迫特征按照特定的顺序进行检查（不利用相对独立的特征分类）
# P265.Ex6-8 计算标签链表的熵
import math


def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in nltk.FreqDist(labels)]
    return -sum([p * math.log(p, 2) for p in probs])


print(entropy(['male', 'male', 'male', 'male']))
print(entropy(['male', 'female', 'male', 'male']))
print(entropy(['male', 'female', 'male', 'female']))

# 5 朴素贝叶斯分类器
# 在朴素贝叶斯分类器中，每个特征都有发言权（避免了决策树中数据量小的特征被忽略的问题）

# 5.1 潜在概率模型：为输入选择最有可能的标签
# 基于假设：每个输入值都是经过先为该输入值选择类标签，然后产生每个特征的方式而产生的，每个特征与其他特征都是完全独立的。

# 5.2 零计数和平滑
# 建立朴素贝叶斯模型时，因为特征过多可能会产生零计数（稀疏数据），需要采用平滑技术，使得每个特征都有计数。
# nltk.probability提供了多种平滑技术支持。

# 5.3 非二元特征的解决办法
# 1）利用装箱技术将之转换为二元特征
# 2）使用回归方法来模拟数字特征的概率

# 5.4 独立的朴素性：朴素（naive，天真）的原因是不切实际地假设所有的特征都是朴素独立的
# 5.5 双重计数的原因：在训练过程中特征的贡献被分开计算，但是当使用分类器为新输入选择标签时，这些特征的贡献就被组合在一起了。

# 6 最大熵分类器：找出能使训练语料的整体似然性最大的参数组。
# 最大熵分类器与朴素贝叶斯分类器使用的模型相似：
# * 朴素贝叶斯分类器使用概率设计模型的参数
# * 最大熵分类器使用搜索技术找出一组能够最大限度地提高分类器性能的参数
# 最大熵分类器使用迭代优化技术选择模型参数

# 6.1 最大熵模型：是朴素贝叶斯模型的泛化。
# 朴素贝叶斯模型为每个标签定义一个参数，指定其先验概率，为每个（特征，标签）对定义一个参数，为标签的似然性指定其独立特征的贡献。
# 最大熵模型让用户来判断使用什么样的参数来组合标签和特征。
# 单独的参数既可以将一个特征和多个标签关联起来，也可以将一个特征与一个给定的标签关联起来。即允许模型“概括”相关的标签或特征之间的差异。
# 每个标签和特征的组合都可以接收其自身的参数，并称之为联合特征。联合特征是加了标签值的属性，而（简单）特征是未加标签值的属性。
# 用来构建最大熵模型的联合特征完全反映了朴素贝叶斯模型所使用的联合特征，即最大熵模型的特征比朴素贝叶斯模型的特征更加复杂，也更加多样。

# 6.2 最大化熵
# 最大熵原理是指在已知的分布下，选择熵最高的分布。

# 6.3 生成式分类器 与 条件式分类器
# 朴素贝叶斯分类器是一个生成式分类器，可以回答以下的问题：
# 1）一个给定的输入的最可能的标签是什么？
# 2）对于一个给定输入，一个给定标签有多大可能性？
# 3）最有可能的输入值是多少？
# 4）一个给定输入值的可能性有多大？
# 5）一个给定输入具有一个给定标签的可能性有多大？
# 6）对于一个可能有两个值的输入，最可能的标签是什么？
# 最大熵分类器是一个条件式分类器
# 条件式分类器建立模型预测：一个给定输入值的标签的概率，只能回答问题1~2
# 生成式模型 比 条件式模型 更加强大，但是计算代价也更大

# 7 为语言模型建模
# 模型可以采用：1）监督式分类技术；2）分析型激励模型
# 模型的目的：1）了解语言模型；2）预测新的语言数据

# 7.1 模型能够告诉我们什么？
# 描述型模型捕获数据中的模式，不提供任何有关数据包含这些模式的原因；提供数据内相关信息
# 解释型模型试图捕获造成语言模式的属性和关系；提供假设因果关系

# 大多数从语料库中自动构建的模型都是描述型模型。可以用来预测未来的数据，不考虑假设的因果关系。

# 8 小结
# 在语料库的语言数据建模，可以帮助理解语言模型，也可以用于预测新的语言数据
# 监督式分类器使用加标签的训练语料库来建立模型，该模型可以基于输入数据的特征，预测该输入数据的标签
# 监督式分类器可以应用于：文档分类、词性标注、语句分割、对话行为类型识别、确定蕴涵关系等等
# 训练监督式分类器时，应该把语料分为3个数据集：训练集、开发测试集和测试集
# 评估监督式分类器时，应该使用训练集和开发测试集中没有使用的数据
# 决策树是自动构建树结构的流程图，用于基于输入数据的特征添加标签。
#   优点是易于解释，缺点是不适合在决策合适标签过程中处理朴素影响的特征值。
# 朴素贝叶斯分类器中每个特征独立地决定应该使用哪个标签。
#   优点是允许特征值间有关系，缺点是当两个或者更多的特征高度相关时将会出现问题。
# 最大熵分类器使用的基本模型与朴素贝叶斯分类器相似
#   优点是不再强调特征独立性假设，并且允许更加复杂的特征组合；
#   缺点是需要通过迭代优化技术寻找使训练集概率最大化的特征权值集合，并且不同的初值会得到不同的优化结果，并且可能只是局部最优。
# 大多数从语料库中自动构建的模型都是描述型模型。只能描述哪些特征与给定的模式或者结构相关，无关给出这些特征与模式之间的因果关系。
