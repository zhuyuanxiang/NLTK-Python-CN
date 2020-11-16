# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0308.py
@Version    :   v0.1
@Time       :   2020-11-16 10:18
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# 3.8 分割(Segmentation)(P121)

# 3.8.1 句分割，断句，Sentence Segmentation(P122)
# 计算布朗语料库中每个句子的平均词数
len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())

# Punkt 句子分割器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = sent_tokenizer.tokenize(text)  # 转为使用Punkt句子分割器
sents[171:181]


# 3.8.2 词分割，分词(Word Segmentation)(P123)
# Ex3-2
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i + 1])
            last = i + 1
    words.append(text[last:])
    return words


text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"

seg1 = "0000000000000001000000000010000000000000000100000000000"
segment(text, seg1)

seg2 = "0100100100100001001001000010100100010010000100010010000"
segment(text, seg2)

words = segment(text, seg2)
text_size = len(words)
lexicon_size = len(' '.join(list(set(words))))
text_size + lexicon_size


# P124 Ex3-3 计算存储词典和重构源文本的成本，计算目标函数，评价分词质量，得分越小越好
def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size


text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"

seg1 = "0000000000000001000000000010000000000000000100000000000"
evaluate(text, seg1)

seg2 = "0100100100100001001001000010100100010010000100010010000"
evaluate(text, seg2)

seg3 = "0000100100000011001000000110000100010000001100010000001"
evaluate(text, seg3)

# P125 Ex3-4 使用模拟退火算法的非确定性搜索；
# 1) 一开始仅搜索短语分词；
# 2) 然后随机扰动0和1，它们与“温度”成比例；
# 3) 每次迭代温度都会降低，扰动边界会减少。

from random import randint


def flip(segs, pos):
    return segs[:pos] + str(1 - int(segs[pos])) + segs[pos + 1:]


def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs) - 1))
    return segs


def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, int(round(temperature)))
            score = evaluate(text, guess)
            if score < best:
                best_segs, best = guess, score
        segs, score = best_segs, best
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs


print("anneal(seg1)= ", anneal(text, seg1, 5000, 1.2))
# 小的得分，不一定是合理的分词结果，说明评价函数存在问题
print("anneal(seg2)= ", anneal(text, seg2, 5000, 1.2))
