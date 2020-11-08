# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0102.py
@Version    :   v0.1
@Time       :   2020-11-08 9:51
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
from nltk.book import *

from tools import *

# Chap1 语言处理与Python
# 目的：
# 1）简单的程序如何与大规模的文本结合？
# 2）如何自动地提取出关键字和词组？如何使用它们来总结文本的风格和内容？
# 3）Python为文本处理提供了哪些工具和技术？
# 4）自然语言处理中还有哪些有趣的挑战？

# 1.5 自然语言理解的自动化处理
# 1.5.1 词意消歧：相同的单词在不同的上下文中指定不同的意思
# 1.5.2 指代消解：检测动词的主语和宾语
# 指代消解：确定代词或名字短语指的是什么？
# 语义角色标注：确定名词短语如何与动词想关联（如代理、受事、工具等）
# 1.5.3 自动生成语言：如果能够自动地解决语言理解问题，就能够继续进行自动生成语言的任务，例如：自动问答和机器翻译。
# 1.5.4 机器翻译：
# 文本对齐：自动配对组成句子
# 1.5.5 人机对话系统：图1-5，简单的话音对话系统的流程架构
# 1.5.6 文本的含义：文本含义识别（Recognizing Textual Entailment, RTE）
# 1.5.7 NLP的局限性

