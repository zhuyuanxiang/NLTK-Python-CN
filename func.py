# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   func.py
@Version    :   v0.1
@Time       :   2020-11-08 9:56
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk


def lexical_diversity(my_text_data):
    """每个单词的平均使用次数"""
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = vocab_size / word_count
    return diversity_score


def percentage(count, total):
    """特定单词在文本中占据的百分比"""
    return 100 * count / total


def plural(word):
    """Ex2-2：生成任意英语名词的复数形式"""
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def unusual_words(text):
    """Ex2-3：过滤文本：计算文本的词汇表，删除所有在现有的词汇列表中出现的元素，只留下罕见的 或者 拼写错误的 词汇"""
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)


def content_fraction(text):
    """计算文本中不包含在停用词列表中的词所占的比例"""
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


def stress(pron):
    """定义提取重音数字的函数"""
    return [
            char
            for phone in pron
            for char in phone
            if char.isdigit()
    ]