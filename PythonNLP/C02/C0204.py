# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0204.py
@Version    :   v0.1
@Time       :   2020-11-13 10:37
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
from func import *
from func import stress
from tools import *

# 2.4 词典资源
# 2.4.1 词汇列表语料库
# 过滤文本，删除掉常见英语词典中的单词，留下罕见的或者拼写错误的词汇

show_subtitle("austen-sense 中的非常用词汇")
print(unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))[:13])
show_subtitle("nps_chat 中的非常用词汇：")
print(unusual_words(nltk.corpus.nps_chat.words())[:13])

# 停止词语料库(stopwords)：包括的是高频词汇，是会使说话停止的单词，不是不再使用的单词
from nltk.corpus import stopwords

show_subtitle("english stopwords")
print(stopwords.words('english'))

# 利用停止词，筛选掉文本中三分之一的单词
print("reuters 中非停用词所占的比例=", content_fraction(nltk.corpus.reuters.words()))

# 图2-6：词谜：在由随机选择的字母组成的网格中，选择里面的字母组成单词。
# 这个谜题叫做「目标」
# 使用 ‘egivrvonl’ 字母可以组成多少个不少于6个字母的单词？
# 确保「r」字母出现，其他字母出现频率不高于 puzzle_letters 的要求
puzzle_letters = nltk.FreqDist('egivrvonl')
puzzle_letters.tabulate()
obligatory = 'r'
word_list = nltk.corpus.words.words()
target_word_list = [
        w
        for w in word_list
        if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters
]
print("target word list= ", target_word_list)

# 使用名字语料库
# 找出同时出现在两个语料库中的名字(即无法分辨性别的名字)
names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
name_list = [
        w
        for w in male_names
        if w in female_names
]
print("name list=", name_list[:13])
print("name list length= ", len(name_list))

show_subtitle("使用 set() 函数计算")
print("name list=", list(set(male_names).difference(set(male_names).difference(female_names)))[:13])
print("name list length= ", len(set(male_names).difference(set(male_names).difference(female_names))))

name_ends = (
        (fileid, name[-2:])
        for fileid in names.fileids()
        for name in names.words(fileid)
)
for i, name_end in enumerate(name_ends):
    if i <= 13:
        print(name_end)

name_ends = (
        (fileid, name[-2:])
        for fileid in names.fileids()
        for name in names.words(fileid)
)
cfd = nltk.ConditionalFreqDist((fileid, name[-2:]) for fileid in names.fileids() for name in names.words(fileid))
cfd.plot()  # 图2-7 显示男性与女性名字的结尾字母
cfd.tabulate()

# 2.4.2 发音词典
entries = nltk.corpus.cmudict.entries()
print("len(entries)= ", len(entries))
for entry in entries[39943:39951]:
    print(entry)

# 寻找词典中发音包含三个音素的条目
for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph1, ph2, ph3)

# 寻找所有与 nicks 发音相似的单词
syllable = ['N', 'IH0', 'K', 'S']
word_list = [
        word
        for word, pron in entries
        if pron[-4:] == syllable
]
print(word_list)

# 寻找拼写以'n'结尾，发音以'M'结尾的单词
word_list = [
        w
        for w, pron in entries
        if pron[-1] == 'M' and w[-1] == 'n'
]
print(word_list)

# 提取重音数字(0-无重音，1-主重音，2-次重音)
word_list = [
        (w, pron)
        for w, pron in entries
        if stress(pron) == ['0', '1', '0', '2', '0']
]
print(word_list[:13])

# 提取重音数字(0-无重音，1-主重音，2-次重音)
word_list = [
        (w, pron)
        for w, pron in entries
        if stress(pron) == ['0', '2', '0', '1', '0']
]
print(word_list[:13])

# 拆分映射、列表、字符串的测试
ex_pron = ('surrealistic', ['S', 'ER0', 'IY2', 'AH0', 'L', 'IH1', 'S', 'T', 'IH0', 'K'])
(word, pron) = ex_pron
show_subtitle(word)
for phone in pron:
    print(phone, end='=')
    for char in phone:
        print(char, end=',')
    print()

# P69 使用条件频率分布寻找相应词汇的最小对比集
# 找到所有 P 开头的三音素词，并比照它们的第一个和最后一个音素来分组
p3 = [
        (pron[0] + '-' + pron[2], word)
        for (word, pron) in entries
        if pron[0] == 'P' and len(pron) == 3
]
cfd = nltk.ConditionalFreqDist(p3)
cfd.tabulate(conditions=['P-P', 'P-R'])
print("cfd['P-P']= ", cfd['P-P'])
for template in cfd.conditions():
    if len(cfd[template]) > 10:
        words = cfd[template].keys()
        wordlist = ' '.join(words)
        print(template, wordlist[:70] + "...")

# 访问词典的方式
prondict = nltk.corpus.cmudict.dict()
print("prondict['fire']= ", prondict['fire'])
# print("prondict['blog'] = ",prondict['blog']  # 词典中没有，报错KeyError
# 词典元素赋值
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
print("prondict['blog']= ", prondict['blog'])

# 在词典中寻找单词的发音
text = ['natural', 'language', 'processing']
pron_list = [
        ph
        for w in text
        for ph in prondict[w][0]
]
print("word pronoun list= ", pron_list)

# 加[0]是因为natural有两个发音，取其中一个就好了
pron_list = [
        ph
        for w in text
        for ph in prondict[w]
]
print("'natural' pronoun list= ", pron_list)
print("prondict['natural']=", prondict['natural'])

# P70 2.4.3 比较词表（Swadesh wordlists）
# 包括几种语言的约200个常用词的列表，可以用于比较两个语言之间的差别，也可以用于不同语言的单词翻译
from nltk.corpus import swadesh

print("swadesh.fileids()= ", swadesh.fileids())
print("swadesh.words('en')= ", swadesh.words('en'))

fr2en = swadesh.entries(['fr', 'en'])
print("fr2en= ", fr2en[:13])
translate = dict(fr2en)
print("translate= ", translate)
print("translate['chien']= ", translate['chien'])

de2en = swadesh.entries(['de', 'en'])
translate.update(dict(de2en))
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(es2en))
print("translate= ", translate)

print("translate['jeter']= ", translate['jeter'])
print("translate['Hund']= ", translate['Hund'])
print("translate['perro']= ", translate['perro'])

languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])

# P71 2.4.4 词汇工具 Toolbox Shoebox，是由一些条目的集合组成，每个条目由一个或多个字段组成，大多数字段都是可选的或者重复的

from nltk.corpus import toolbox

# 罗托卡特语(Rotokas) 的词典
# 第一个条目(kaa)，表示「窒息」
rotokas = toolbox.entries('rotokas.dic')
for i, word in enumerate(rotokas):
    if i <= 13:
        print(word)
