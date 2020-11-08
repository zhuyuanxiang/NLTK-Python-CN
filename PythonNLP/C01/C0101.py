# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0101.py
@Version    :   v0.1
@Time       :   2020-11-08 9:51
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
from nltk.book import *

from func import *
from tools import *

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1.1 语言计算：文本和词汇
# 1.1.3 搜索文本

show_title("搜索单词 monstrous，显示其所在的上下文")
text1.concordance('monstrous')

show_title("搜索与 monstrous 具有相同上下文的单词")
show_subtitle("text1")
text1.similar('monstrous')
show_subtitle("text2")
text2.similar('monstrous')

show_title("搜索 monstrous 与 very 两个单词的相同上下文")
text2.common_contexts(['monstrous', 'very'])

# 图1-2：美国总统就职演说词汇分布图：可以用来研究随时间推移语言使用上的变化
text4.dispersion_plot(['citizens','democracy','freedom','duties','America'])

# NLTK 3.0的版本已经放弃了这个功能
text3.generate()

# 1.1.4 计数词汇

print("text3 中的单词个数：", len(text3))
print("text3 中的单词数：", len(set(text3)))
print("text3 中的单词排序：", sorted(set(text3))[0:20])

print("text3 中每个单词被使用的次数的平均数：", len(text3) / len(set(text3)))
print("text3 中每个单词被使用的次数的平均数：", lexical_diversity(text3))
print("text5 中每个单词被使用的次数的平均数：", lexical_diversity(text5))

print("text3 中单词 'smote' 的使用次数：", text3.count("smote"))
print("text3 中单词 'smote' 的出现比例：", percentage(text3.count("smote"), len(text3)))

print("text4 中单词 'a' 的出现比例：", 100 * text4.count('a') / len(text4))
print("text4 中单词 'a' 的出现比例：", percentage(text4.count('a'), len(text4)))
