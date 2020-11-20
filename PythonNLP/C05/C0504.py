# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0504.py
@Version    :   v0.1
@Time       :   2020-11-20 11:08
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk
import pylab

from nltk.corpus import brown

# 5.4 自动标注（利用不同的方式给文本自动添加词性标记）

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
brown_tagged_words = brown.tagged_words(categories='news')
brown_words = brown.words(categories='news')

# 5.4.1 默认标注器
# 寻找在布朗语料库中新闻类文本使用次数最多的标记
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()

# 因为 'NN' 是使用次数最多的标记，因此设置它为默认标注
raw = 'I do not lie green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
default_tagger.evaluate(brown_tagged_sents)  # 评测默认标注的正确率

# 5.4.2 正则表达式标注器
patterns = [
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # simple past
        (r'.*es$', 'VBZ'),  # 3rd singular present
        (r'.*ould$', 'MD'),  # modals
        (r'.*\'s$', 'NN$'),  # possessive nouns
        (r'.*s$', 'NNS'),  # plural nouns
        (r'(a|an)', 'AT'),
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')  # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])  # 是标注的文本
regexp_tagger.evaluate(brown_tagged_sents)  # brown_tagged_sents 是测试集

# 5.4.3 查询标注器
# 找出 100 个最频繁的词，存储它们最有可能的标记，然后使用这个信息作为“查找标注器”的模型
fd = nltk.FreqDist(brown_words)
cfd = nltk.ConditionalFreqDist(brown_tagged_words)
most_freq_words = fd.most_common(100)
likely_tags = dict(
        (word, cfd[word].max())
        for (word, _) in most_freq_words
)
print("cfd['the']= ", cfd['the'])

# 一元语法模型，统计词料库中每个单词标注最多的词性作为一元语法模型的建立基础
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

sent = brown_sents[3]
baseline_tagger.tag(sent)

# 对于一元语法模型不能标注的单词，使用默认标注器，这个过程叫做“回退”。
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
baseline_tagger.evaluate(brown_tagged_sents)


# Ex5-4 查找标注器的性能评估
def performance(cfd, wordlist):
    lt = dict(
            (word, cfd[word].max())
            for word in wordlist
    )
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown_tagged_sents)


def display():
    word_freqs = nltk.FreqDist(brown_words).most_common()
    words_by_freq = [
            w
            for (w, _) in word_freqs
    ]
    cfd = nltk.ConditionalFreqDist(brown_tagged_words)
    sizes = 2 ** pylab.arange(15)
    # 单词模型容量的大小对性能的影响
    perfs = [
            performance(cfd, words_by_freq[:size])
            for size in sizes
    ]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()


display()
