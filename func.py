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


def lexical_diversity(text):
    """每个单词的平均使用次数"""
    return len(text) / len(set(text))


def percentage(count, total):
    """特定单词在文本中占据的百分比"""
    return 100 * count / total
