# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0205.py
@Version    :   v0.1
@Time       :   2020-11-13 10:37
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :
@Desc       :
@理解：
"""
# P72 2.5 WordNet：面向语言的英语词典，可以寻找同义词
# 2.5.1 单词的 意义 和 同义
from nltk.corpus import wordnet as wn

from tools import *

show_title("synsets() 查找同义词集的集合")
show_subtitle("motocar 没有同义词集的集合")
print("wn.synsets('motocar')= ", wn.synsets('motocar'))

# 以下单词的定义中都含有'car.n.01'
print("wn.synsets('motorcar')= ", wn.synsets('motorcar'))
print("wn.synsets('car')= ", wn.synsets('car'))
print("wn.synsets('auto')= ", wn.synsets('auto'))
print("wn.synsets('automobile')= ", wn.synsets('automobile'))
print("wn.synsets('machine')= ", wn.synsets('machine'))

show_title("synset() 查找同义词集；synsets() 查找同义词集的集合")
show_subtitle("对单词查找同义词集，返回 ValueError")
# print("wn.synset('car')= ", wn.synset('car'))
show_subtitle("对单词查找同义词集的集合")
print("wn.synsets('car')= ", wn.synsets('car'))
show_subtitle("对同义词定义查找同义词集")
print("wn.synset('car.n.01')= ", wn.synset('car.n.01'))
show_subtitle("对同义词定义查找同义词集的集合")
print("wn.synsets('car.n.01')= ", wn.synsets('car.n.01'))

show_title("查找词元集合")
show_subtitle("对单词查找词元集合")
print("wn.lemmas('car')= ", wn.lemmas('car'))
show_subtitle("对同义词定义查找词元集合")
print("wn.lemmas('car.n.01')= ", wn.lemmas('car.n.01'))
show_subtitle("同义词集的词元集合")
print("wn.synset('car.n.01').lemmas()= ", wn.synset('car.n.01').lemmas())
show_subtitle("同义词集的词元集合中对应的单词集合")
print("wn.synset('car.n.01').lemma_names()= ", wn.synset('car.n.01').lemma_names())

# 先查找单词的同义词定义集合，再对同义词定义查找相应的词元集合对应的单词集合
for synset in wn.synsets('car'):
    print(synset, '=', synset.lemma_names())

show_subtitle("单词的含义")
print("wn.synset('car.n.01').definition()= ", wn.synset('car.n.01').definition())

show_subtitle("单词的句子样例")
print("wn.synset('car.n.01').examples()= ", wn.synset('car.n.01').examples())

show_subtitle("对同义词定义查找词元集合")
print("wn.lemma('car.n.01.automobile')= ", wn.lemma('car.n.01.automobile'))
show_subtitle("对同义词定义查找同义词集")
print("wn.lemma('car.n.01.automobile').synset= ", wn.lemma('car.n.01.automobile').synset())
show_subtitle("对同义词定义查找对应的单词")
print("wn.lemma('car.n.01.automobile').name= ", wn.lemma('car.n.01.automobile').name())

# P74 2.5.2 WordNet的层次结构：
# 图2-8：WordNet 概念的层次
# -   每个节点对应一个同义词集
# -   边表示上位词/下位词关系，即上级概念 与 下级概念 的关系
# 根同义词集-->下位词-->上位词
motorcar = wn.synset('car.n.01')
print("motorcar 上位词= ", motorcar.hypernyms())
types_of_motorcar = motorcar.hyponyms()
print("len(motorcar 下位词)= ", len(types_of_motorcar))
show_subtitle("motorcar 下位词列表")
for motorcar_hyponyms in types_of_motorcar:
    print(motorcar_hyponyms)

lemma_of_motorcar = sorted([
        lemma.name()
        for synset in types_of_motorcar
        for lemma in synset.lemmas()
])
len(lemma_of_motorcar)

# 到根结点的路径，可能会有多条，例如“汽车”被归类为“车辆”和“容器”
motorcar = wn.synset('car.n.01')
paths = motorcar.hypernym_paths()
print("len(paths)= ", len(paths))
show_subtitle("通过hypernym_paths抵达motorcar的路径")
for path in paths:
    print(path)

show_subtitle("motorcar 的根节点")
print(motorcar.root_hypernyms())
show_subtitle("所有事物的根节点都是'entity.n.01'")
wn.synset('love.n.01').root_hypernyms()

nltk.app.wordnet()  # ToDo: 在 Console 环境下 和 Notebook 环境下都不能正常使用

# P77 2.5.3 更多的词汇关系
# 上位词 和 下位词 之间的关系被称为词汇关系，因为它们之间是同义集关系。
# 部分 和 整体 之间的关系也被称为词汇关系，因为它们之间是包含和从属的关系。例如：树 与 树叶
show_subtitle("部分-整体关系。树由树桩、树干、树冠、枝干、树节组成")
print("wn.synset('tree.n.01').part_holonyms()= ", wn.synset('tree.n.01').part_holonyms())
print("wn.synset('tree.n.01').part_meronyms()= ", wn.synset('tree.n.01').part_meronyms())
print("wn.synset('burl.n.02').part_holonyms()= ", wn.synset('burl.n.02').part_holonyms())

show_subtitle("实质关系。树的实质是心材和边材")
print("wn.synset('tree.n.01').substance_holonyms()= ", wn.synset('tree.n.01').substance_holonyms())
print("wn.synset('tree.n.01').substance_meronyms()= ", wn.synset('tree.n.01').substance_meronyms())
print("wn.synset('heartwood.n.01').substance_holonyms()= ", wn.synset('heartwood.n.01').substance_holonyms())

show_subtitle("集合关系。森林由树木和丛林组成")
print("wn.synset('tree.n.01').member_holonyms()= ", wn.synset('tree.n.01').member_holonyms())
print("wn.synset('forest.n.01').member_meronyms()= ", wn.synset('forest.n.01').member_meronyms())

for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())

print("wn.synset('mint.n.01')= ", wn.synset('mint.n.01'))
print("wn.synset('mint.n.01').definition()= ", wn.synset('mint.n.01').definition())
print("wn.synset('batch.n.02').lemma_names()= ", wn.synset('batch.n.02').lemma_names())

print("wn.synset('mint.n.02')= ", wn.synset('mint.n.02'))
print("wn.synset('mint.n.04').part_holonyms()= ", wn.synset('mint.n.04').part_holonyms())
print("wn.synset('mint.n.04').part_meronyms()= ", wn.synset('mint.n.04').part_meronyms())

print("wn.synset('mint.n.05')= ", wn.synset('mint.n.05'))
print("wn.synset('mint.n.04').substance_holonyms()= ", wn.synset('mint.n.04').substance_holonyms())
print("wn.synset('mint.n.04').substance_meronyms()= ", wn.synset('mint.n.04').substance_meronyms())

print("wn.synset('mint.n.04').member_holonyms()= ", wn.synset('mint.n.04').member_holonyms())
print("wn.synset('mint.n.04').member_meronyms()= ", wn.synset('mint.n.04').member_meronyms())

show_subtitle("蕴涵: entailments()")
print("wn.synset('mint.n.04').entailments()= ", wn.synset('mint.n.04').entailments())

# 动词之间的关系
print("wn.synset('walk.v.01').entailments()= ", wn.synset('walk.v.01').entailments())
print("wn.synset('eat.v.01').entailments()= ", wn.synset('eat.v.01').entailments())
print("wn.synset('tease.v.03').entailments()= ", wn.synset('tease.v.03').entailments())

show_subtitle("反义词: antonyms()")
# 不能通过同义词集寻找反义词
print("wn.synset('supply.n.02.supply').antonyms()= ", wn.synset('supply.n.02.supply').antonyms())
print("wn.lemma('supply.n.02.supply').antonyms()= ", wn.lemma('supply.n.02.supply').antonyms())
print("wn.lemma('rush.v.01.rush').antonyms()= ", wn.lemma('rush.v.01.rush').antonyms())
print("wn.lemma('horizontal.a.01.horizontal').antonyms()= ", wn.lemma('horizontal.a.01.horizontal').antonyms())
print("wn.lemma('staccato.r.01.staccato').antonyms()= ", wn.lemma('staccato.r.01.staccato').antonyms())

# 5.4. 语义相似度（拥有共同的上位词的同义词集之间的距离）
right = wn.synset('right_whale.n.01')

minke = wn.synset('minke_whale.n.01')
show_subtitle("right 与 minke 的相同上位词")
print(right.lowest_common_hypernyms(minke))
print("wn.synset('baleen_whale.n.01').min_depth()= ", wn.synset('baleen_whale.n.01').min_depth())

show_subtitle("right 与 orca 的相同上位词")
orca = wn.synset('orca.n.01')
print(right.lowest_common_hypernyms(orca))
print("wn.synset('whale.n.02').min_depth()= ", wn.synset('whale.n.02').min_depth())

show_subtitle("right 与 tortoise 的相同上位词")
tortoise = wn.synset('tortoise.n.01')
print(right.lowest_common_hypernyms(tortoise))
print("wn.synset('vertebrate.n.01').min_depth()= ", wn.synset('vertebrate.n.01').min_depth())

show_subtitle("right 与 novel 的相同上位词")
novel = wn.synset('novel.n.01')
print(right.lowest_common_hypernyms(novel))
print("wn.synset('entity.n.01').min_depth()= ", wn.synset('entity.n.01').min_depth())

show_subtitle("路径相似度度量")
print("right.path_similarity(minke)= ",right.path_similarity(minke))
print("right.path_similarity(orca)= ",right.path_similarity(orca))
print("right.path_similarity(tortoise)= ",right.path_similarity(tortoise))
print("right.path_similarity(novel)= ",right.path_similarity(novel))

# 2.6 小结
# 文本语料库是一个大型结构化文本的集合
# 条件频率分布是频率分布的集合，每个分布都有不同的条件
# WordNet是一个面向语义的英文词典，由同义词的集合——或称为同义词集(synsets)——组成，并且组成一个网络
