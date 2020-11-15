# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0304.py
@Version    :   v0.1
@Time       :   2020-11-15 10:19
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import re

from tools import *

# 3.4. 使用正则表达式检测词组搭配
# （本书可以帮助你快速了解正则表达式，）
show_subtitle("取出所有的小写字母拼写的单词")
wordlist = [
        w
        for w in nltk.corpus.words.words('en')
        if w.islower()
]
print(wordlist[:13])

# 3.4.1 使用基本的元字符
show_subtitle("搜索以'ed'结尾的单词")
word_ed_list = [
        w
        for w in wordlist
        if re.search(r'ed$', w)
]
print(word_ed_list[:13])

show_subtitle("搜索以'**j**t**'形式的单词，'^'表示单词的开头，'$'表示单词的结尾")
word_jt_list = [
        w
        for w in wordlist
        if re.search(r'^..j..t..$', w)
]
print(word_jt_list[:13])

# 3.4.2 范围与闭包
show_subtitle("[ghi]表示三个字母中任意一个")
word_ghi_list = [
        w
        for w in wordlist
        if re.search(r'^[ghi][mno][jlk][def]$', w)
]
print(word_ghi_list[:13])

chat_words = sorted(set(
        w
        for w in nltk.corpus.nps_chat.words()
))
print(chat_words[:13])

show_subtitle("'+'表示一个或者多个")
word_plus_list = [
        w
        for w in chat_words
        if re.search(r'^m+i+n+e+$', w)
]
print(word_plus_list[:13])

show_subtitle("'*'表示零个或者多个")
word_star_list = [
        w
        for w in chat_words
        if re.search(r'^m*i*n*e*$', w)
]
print(word_star_list[:13])

# [^aeiouAEIOU]：表示没有这些字母的单词，即没有元音字母的单词，就是标点符号
show_subtitle("'^'表示没有这些字符")
word_hat_list = [
        w
        for w in chat_words
        if re.search(r"[^aeiouAEIOU]", w)
]
print(word_hat_list[:13])

wsj = sorted(set(nltk.corpus.treebank.words()))
# 前面两个输出一样的结果，因为小数肯定都会有整数在前面，而第三个不一样，是因为'?'表示零个或者一个，不包括大于10的整数
show_subtitle("对比 * + ? 三个的区别")
print("*= ", len([w for w in wsj if re.search(r"^[0-9]*\.[0-9]+$", w)]))
print("+= ", len([w for w in wsj if re.search(r"^[0-9]+\.[0-9]+$", w)]))
print("?= ", len([w for w in wsj if re.search(r"^[0-9]?\.[0-9]+$", w)]))

show_subtitle("'\\' 斜杠的作用")
word_slash_list = [
        w
        for w in wsj
        if re.search(r"^[A-Z]+\$$", w)
]
print(word_slash_list)

show_subtitle("四位数的整数")
word_four_list = [
        w
        for w in wsj
        if re.search(r"^[0-9]{4}$", w)
]
print(word_four_list)

show_subtitle("1位以上整数-3~5位长的单词")
word_num_word_list = [
        w
        for w in wsj
        if re.search(r"^[0-9]+-[a-z]{3,5}$", w)
]
print(word_num_word_list)

show_subtitle("5位以上长的单词-2~3位长的单词-6位以下长的单词")
word_word_word_list = [
        w
        for w in wsj
        if re.search(r"^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$", w)
]
print(word_word_word_list)

show_subtitle("寻找分词")
word_participle_list = [
        w
        for w in wsj
        if re.search(r"(ed|ing)$", w)
]
print(word_participle_list[:13])

# 表3-3 正则表达式的基本元字符，其中包括通配符、范围和闭包 P109
# 原始字符串（raw string）：给字符串加一个前缀“r”表明后面的字符串是个原始字符串。
