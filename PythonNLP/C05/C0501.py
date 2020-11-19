# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0501.py
@Version    :   v0.1
@Time       :   2020-11-19 9:48
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk
from nltk import word_tokenize
from nltk.corpus import brown

from tools import show_subtitle

# Ch5 分类和标注词汇

# 1.  什么是词汇分类，在自然语言处理中它们如何使用？
# 2.  对于存储词汇和它们的分类来说什么是好的 Python 数据结构？
# 3.  如何自动标注文本中每个词汇的词类？

# -   词性标注（parts-of-speech tagging，POS tagging）：简称标注。将词汇按照它们的词性（parts-of-speech，POS）进行分类并对它们进行标注
# -   词性：也称为词类或者词汇范畴。
# -   标记集：用于特定任务标记的集合。

brown_words = brown.words(categories='news')
brown_tagged_words = brown.tagged_words(categories='news')
brown_sents = brown.sents(categories='news')
brown_tagged_sents = brown.tagged_sents(categories='news')

# Sec 5.1 使用词性标注器
text = word_tokenize("And now for something completely different")
nltk.pos_tag(text)
nltk.help.upenn_tagset('CC')
nltk.help.upenn_tagset('RB')
nltk.help.upenn_tagset('IN')
nltk.help.upenn_tagset('NN')
nltk.help.upenn_tagset('JJ')
nltk.corpus.brown.readme()
print(nltk.corpus.gutenberg.readme())

# 处理同形同音异义词，系统正确标注了
# 前面的refUSE是动词，后面的REFuse是名词
# 前面的permit是动词，后面的permit是名字
text = word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)
text = word_tokenize("They refuse to permit us to obtain the beautiful book")
nltk.pos_tag(text)

# 找出形如w1 w w2的上下文，然后再找出所有出现在相同上下文的词 w'，即w1 w' w2
# 用于寻找相似的单词，因为这些单词处于相同的上下文中
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
show_subtitle("text.similar('word')")
text.similar('word')
show_subtitle("text.similar('woman')")
text.similar('woman')
show_subtitle("text.similar('bought')")
text.similar('bought')
show_subtitle("text.similar('over')")
text.similar('over')
show_subtitle("text.similar('the')")
text.similar('the')
