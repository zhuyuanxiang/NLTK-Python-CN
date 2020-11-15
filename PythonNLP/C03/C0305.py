# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0305.py
@Version    :   v0.1
@Time       :   2020-11-15 11:11
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import re

from tools import *

# 3.5. 正则表达式的有益应用
# 3.5.1 提取字符块
show_subtitle("P109 提取元音字符块")
word = 'supercalifragilisticexpialidocious'
word_piece_list = re.findall(r'[aeiou]', word)
print(word_piece_list, "长度= ", len(word_piece_list))

show_subtitle("P109 提取两个元音字符块")
wsj = sorted(set(nltk.corpus.treebank.words()))
word_pieces_list = [
        vs
        for word in wsj
        for vs in re.findall(r'[aeiou]{2,}', word)
]
print(word_pieces_list)

show_subtitle("统计双元音字符块的数目")
fd = nltk.FreqDist(word_pieces_list)
fd.most_common(12)

show_subtitle("提取日期格式中的整数值")
numbers_list = [
        int(n)
        for n in re.findall(r'[0-9]+', '2009-12-31')
]
print(numbers_list)

# 3.5.2 在字符块上做更多的事情
# 使用findall()完成更加复杂的任务
show_subtitle("P110 忽略掉单词内部的元音")
# 第一个模板保证元音在首字母和元音在尾字母的依然保留
regexp1 = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
# 第二个模板会删除所有元音字母
regexp2 = r'[^AEIOUaeiou]'


def compress(word, regexp):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)


english_udhr = nltk.corpus.udhr.words('English-Latin1')

english_tmp1 = [
        compress(w, regexp1)
        for w in english_udhr
]
print("english_tmp1= ", english_tmp1[:13])
print("len(english_tmp1)= ", len(english_tmp1))

english_tmp2 = [
        compress(w, regexp2)
        for w in english_udhr
]
print("english_tmp2= ", english_tmp2[:13])
print("len(english_tmp2)= ", len(english_tmp2))

print("english_udhr[:75]= ", nltk.tokenwrap(english_tmp1[:75]))

show_subtitle("P111 提取「辅音-元音序列对」，并且统计单词库中这样的序列对的数目")
rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [
        cv
        for w in rotokas_words
        for cv in re.findall(r'[ptksvr][aeiou]', w)
]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()

show_subtitle("定义「辅音-元音序列对」所对应的单词集合")
cv_word_pairs = [
        (cv, w)
        for w in rotokas_words
        for cv in re.findall(r'[ptksvr][aeiou]', w)
]
cv_index = nltk.Index(cv_word_pairs)
print("cv_index['su']= ", cv_index['su'])
print("cv_index['po']= ", cv_index['po'])


# 3.5.3 查找词干
# P112 查找词干(忽略词语结尾，只处理词干）
def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word


# 只取出词尾（只提取了后缀，没有提出词干）
regexp = r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$'
print(re.findall(regexp, 'processing'))

# 输出了整个单词（提取符合后缀的字符串，"(?:)"的作用，但是没有提取出词干）
regexp = r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$'
print(re.findall(regexp, 'processing'))  # 符合词缀要求的单词可以提取出来
print(re.findall(regexp, 'processooo'))  # 不符合词缀要求的单词就不提取出来

# 将单词分解为词干和后缀
regexp = r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$'
print(re.findall(regexp, 'processing'))
print(re.findall(regexp, 'processes'))  # 使用贪婪匹配模式，错误分解单词

# 不使用贪婪匹配模式
regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$'
print(re.findall(regexp, 'processes'))
print(re.findall(regexp, 'process'))  # 需要单词背景知识，将这类单词剔除，否则会错误地提取词干
print(re.findall(regexp, 'language'))  # 没有单词背景知识时，如果对于没有词缀的单词会无法提取出单词来

# 正确处理没有后缀的单词
regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
print(re.findall(regexp, 'language'))


# 更加准确地词干提取模板，先将原始数据分词，然后提取分词后的词干
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem


raw = """DENNIS: Listen,
strange women lying in ponds distributing swords
is no basis for a system of government.
Supreme executive power derives from a mandate from the masses, 
not from some farcical aquatic ceremony."""

tokens = nltk.word_tokenize(raw)
stem_list = [
        stem(t)
        for t in tokens
]
print(stem_list)

# 正则表达式的展示函数nltk.re_show()，可以把符合正则表达式要求的字符标注出来
# 不能使用re.findall()中的正则表达式标准。需要使用基本的正则表达式标准。
# P109 表3-3 正则表达式基本元字符，P120 表3-4 正则表达式符号
# 也可以参考《自然语言处理综论（第二版）》P18

regexp = r'[ing|ly|ed|ious|ies|ive|es|s|ment]$'
nltk.re_show(regexp, raw)

regexp = r'(ing)$'
nltk.re_show(regexp, raw)

regexp = r'[ing]$'
nltk.re_show(regexp, raw)

regexp = r'ing$'
nltk.re_show(regexp, raw)

regexp = '^[D|s|i|S|n]'
nltk.re_show(regexp, raw)  # '^' 表示行的开头

regexp = '^[DsiSn]'
nltk.re_show(regexp, raw)  # '[]' 内，用不用|都表示析取

regexp = '[s|.|,]$'
nltk.re_show(regexp, raw)  # '$' 表示行的结尾

regexp = 'ing|tive'
nltk.re_show(regexp, raw)  # '|' 表示析取指定的字符串

regexp = '(ing|tive)'
nltk.re_show(regexp, raw)  # '()' 表示操作符的范围

regexp = '(s){1,2}'
nltk.re_show(regexp, raw)  # '{}' 表示重复的次数

# 3.5.4 搜索已经分词的文本
# P114 对已经实现分词的文本（Text）进行搜索（findall）
from nltk.corpus import gutenberg, nps_chat

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
tokens = moby.tokens
print(tokens[:13])

regexp = r"(?:<a> <.*> <man>)"
moby.findall(regexp)

# 找出文本中"a <word> man"中的word
regexp = r"<a>(<.*>)<man>"
moby.findall(regexp)

regexp = 'ly|ed|ing'
nltk.re_show(regexp, ' '.join(tokens[:75]))

regexp = 'see [a-z]+ now'
nltk.re_show(regexp, ' '.join(tokens[:200]))

chat = nltk.Text(nps_chat.words())
tokens = chat.tokens
print(tokens[:13])

regexp = r"<.*><.*><bro>"
chat.findall(regexp)

regexp = r"<l.*>{3,}"
chat.findall(regexp)

regexp = 'l.+'
nltk.re_show(regexp, ' '.join(tokens[:200]))

regexp = 'h.+'
nltk.re_show(regexp, ' '.join(tokens[200:1000]))

# 正则表达式的测试界面
nltk.app.nemo()

from nltk.corpus import brown

hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
print(hobbies_learned[:13])

regexp = r"<\w*> <and> <other> <\w*s>"
hobbies_learned.findall(regexp)

regexp=r"<as><\w*><as><\w*>"
hobbies_learned.findall(regexp)