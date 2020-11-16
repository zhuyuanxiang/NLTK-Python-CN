# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0307.py
@Version    :   v0.1
@Time       :   2020-11-16 9:14
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import re

import nltk

# 3.7. 用正则表达式为文本分词(P118)
# 分词（Tokenization）：是将字符串切割成可以识别的构成语言数据的语言单元。
# 3.7.1 分词的简单方法
from tools import show_subtitle

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful 
tone though), 'I won't have any pepper in my kitchen AT ALL. Soup does very 
well without--Maybe it's always pepper that makes people hot-tempered,'..."""

print(re.split(r' ', raw))  # 利用空格分词，没有去除'\t'和'\n'
print(re.split(r'[ \t\n]+', raw))  # 利用空格、'\t'和'\n'分词，但是不能去除标点符号
print(re.split(r'\s', raw))  # 使用re库内置的'\s'（匹配所有空白字符）分词，但是不能去除标点符号
print(re.split(r'\W+', raw))  # 利用所有字母、数字和下划线以外的字符来分词，但是将“I'm”、“won't”这样的单词拆分了
print(re.findall(r'\w+|\S\w*', raw))  # 使用findall()分词，可以将标点保留，不会出现空字符串
print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))  # 利用规则使分词更加准确

# 3.7.2 NLTK 的正则表达式分词器(P120)

text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)    # set flag to allow verbose regexps
    (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
  | \w+(?:-\w+)*        # words with optional internal hyphens
  | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
  | \.\.\.            # ellipsis
  | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''
nltk.regexp_tokenize(text, pattern)

# “(?x)”为pattern中的“verbose”标志，将pattern中的空白字符和注释都去掉。
print("'(?x)'= ", nltk.regexp_tokenize(text, '(?x)'))

print("'([A-Z]\.)'= ", nltk.regexp_tokenize(text, '([A-Z]\.)'))
print("'([A-Z]\.)+'= ", nltk.regexp_tokenize(text, '([A-Z]\.)+'))
print("'(?:[A-Z]\.)+'= ", nltk.regexp_tokenize(text, '(?:[A-Z]\.)+'))

print("'\w'= ", nltk.regexp_tokenize(text, '\w'))
print("'\w+'= ", nltk.regexp_tokenize(text, '\w+'))
print("'\w(\w)'= ", nltk.regexp_tokenize(text, '\w(\w)'))  # 每连续两个单词标准的字母，取后面那个字母
print("'\w+(\w)'= ", nltk.regexp_tokenize(text, '\w+(\w)'))  # 每个单词，取最后那个字母
print("'\w(-\w)'= ", nltk.regexp_tokenize(text, '\w(-\w)'))
print("'\w+(-\w)'= ", nltk.regexp_tokenize(text, '\w+(-\w)'))
print("'\w(-\w+)'= ", nltk.regexp_tokenize(text, '\w(-\w+)'))
print("'\w+(-\w+)'= ", nltk.regexp_tokenize(text, '\w+(-\w+)'))
print("'\w(-\w+)*'= ", nltk.regexp_tokenize(text, '\w(-\w+)*'))
print("'\w+(-\w+)*'= ", nltk.regexp_tokenize(text, '\w+(-\w+)*'))

print("'\w+(?:)'))= ", nltk.regexp_tokenize(text, '\w+(?:)'))
print("'\w+(?:)+'))= ", nltk.regexp_tokenize(text, '\w+(?:)+'))
print("'\w+(?:\w)'))= ", nltk.regexp_tokenize(text, '\w+(?:\w)'))
print("'\w+(?:\w+)'))= ", nltk.regexp_tokenize(text, '\w+(?:\w+)'))
print("'\w+(?:\w)*'))= ", nltk.regexp_tokenize(text, '\w+(?:\w)*'))
print("'\w+(?:\w+)*'))= ", nltk.regexp_tokenize(text, '\w+(?:\w+)*'))

print("'\.\.\.'= ", nltk.regexp_tokenize(text, '\.\.\.'))
print("'\.\.\.|([A-Z]\.)+'= ", nltk.regexp_tokenize(text, '\.\.\.|([A-Z]\.)+'))

# (?:) 非捕捉组用法对比
inputStr = "hello 123 world 456 nihao 789"
rePatternAllCapturingGroup = "\w+ (\d+) \w+ (\d+) \w+ (\d+)"
rePatternWithNonCapturingGroup = "\w+ (\d+) \w+ (?:\d+) \w+ (\d+)"
show_subtitle(rePatternAllCapturingGroup)
nltk.regexp_tokenize(inputStr, rePatternAllCapturingGroup)
show_subtitle(rePatternWithNonCapturingGroup)
nltk.regexp_tokenize(inputStr, rePatternWithNonCapturingGroup)

# 3.7.3 进一步讨论分词
# 分词：比预期更为艰巨，没有任何单一的解决方案可以在所有领域都行之有效。
# 在开发分词器时，访问已经手工飘游好的原始文本则理有好处，可以将分词器的输出结果与高品质(也叫「黄金标准」)的标注进行比较。
