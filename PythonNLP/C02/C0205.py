import nltk

# P72 5. WordNet：面向语言的英语词典，可以寻找同义词
# 5.1. 单词的 意义 和 同义
from nltk.corpus import wordnet as wn

wn.synsets('motocar')  # 错误的单词无法查出
# 以下单词的定义中都含有'car.n.01'
wn.synsets('motorcar')
wn.synsets('car')
wn.synsets('auto')
wn.synsets('automobile')
wn.synsets('machine')

wn.synset('car')  # 不能对单词查找同义词集
wn.synset('car.n.01')  # 只能对单词的同义词定义查找同义词集，返回的是同义词集
wn.synsets('car')  # 可以对单词查找同义词集的集合
wn.synsets('car.n.01')  # 对同义词定义查找同义词集的集合为空
wn.lemmas('car')  # 可以对单词查找词元集合
wn.lemmas('car.n.01')  # 对同义词定义查找词元集合为空
wn.synset('car.n.01').lemmas()  # 对同义词定义查找词元集合
wn.synset('car.n.01').lemma_names()  # 对同义词定义查找词元集合中对应的单词集合
# 先查找单词的同义词定义集合，再对同义词定义查找相应的词元集合对应的单词集合
for synset in wn.synsets('car'):
    print(synset, synset.lemma_names())

wn.synset('car.n.01').definition()  # 单词的含义
wn.synset('car.n.01').examples()  # 单词的句子样例

wn.lemma('car.n.01.automobile')  # 对同义词定义查找词元
wn.lemma('car.n.01.automobile').synset()  # 对同义词定义查找同义词集
wn.lemma('car.n.01.automobile').name()  # 对同义词定义查找对应的单词

wn.synset('dish')
wn.synsets('dish')

# P74 WordNet的层次结构：根同义词集-->下位词-->上位词
motorcar = wn.synset('car.n.01')
motorcar.hypernyms()  # 上位词
motorcar.hyponyms()  # 下位词

types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[26]
len(types_of_motorcar)

lemma_of_motorcar = sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()])
len(lemma_of_motorcar)

paths = motorcar.hypernym_paths()  # 到根结点的路径，可能会有多条，例如“汽车”被归类为“车辆”和“容器”
len(paths)

[synset.name() for synset in paths[0]]
[synset.name() for synset in paths[1]]

motorcar.root_hypernyms()  # 根节点
wn.synset('love.n.01').root_hypernyms()  # 所有事物的根节点都是'entity.n.01'吗？

nltk.app.wordnet()  # 可能是因为没有用Jupyter，所以无法显示

# P77 5.3. 更多的词汇关系
# 上位词 和 下位词 之间的关系被称为词汇关系，因为它们之间是同义集关系。
# 部分 和 整体 之间的关系也被称为词汇关系，因为它们之间是包含和从属的关系。例如：树 与 树叶
wn.synset('tree.n.01').part_holonyms()
wn.synset('tree.n.01').part_meronyms()  # 部分关系。树由树桩、树干、树冠、枝干、树节组成。
wn.synset('tree.n.01').substance_holonyms()
wn.synset('tree.n.01').substance_meronyms()  # 实质关系。树的实质是心材和边材
wn.synset('tree.n.01').member_holonyms()  # 集合关系。树组成森林。
wn.synset('forest.n.01').member_meronyms()  # 集合关系。森林由树木和丛林组成。

for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name() + ':', synset.definition())

wn.synset('mint.n.01')
wn.synset('mint.n.01').definition()  # 'mint'的第一个同义词含义就是批量的意思
wn.synset('batch.n.02').lemma_names()

wn.synset('mint.n.04').part_holonyms()  # [Synset('mint.n.02')]
wn.synset('mint.n.04').part_meronyms()
wn.synset('mint.n.04').substance_holonyms() # [Synset('mint.n.05')]
wn.synset('mint.n.04').substance_meronyms()
wn.synset('mint.n.04').member_holonyms()
wn.synset('mint.n.04').member_meronyms()
wn.synset('mint.n.04').entailments()

# 动词之间的关系
wn.synset('walk.v.01').entailments()    # 蕴涵
wn.synset('eat.v.01').entailments()
wn.synset('tease.v.03').entailments()

# 反义词
wn.synset('supply.n.02.supply').antonyms()  # 不能通过同义词集寻找反义词
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms()
wn.lemma('horizontal.a.01.horizontal').antonyms()
wn.lemma('staccato.r.01.staccato').antonyms()

# 5.4. 语义相似度（拥有共同的上位词的同义词集之间的距离）
right = wn.synset('right_whale.n.01')

minke = wn.synset('minke_whale.n.01')
right.lowest_common_hypernyms(minke)    # 相同的上位词

orca = wn.synset('orca.n.01')
right.lowest_common_hypernyms(orca)

tortoise = wn.synset('tortoise.n.01')
right.lowest_common_hypernyms(tortoise)

novel = wn.synset('novel.n.01')
right.lowest_common_hypernyms(novel)

wn.synset('baleen_whale.n.01').min_depth()
wn.synset('whale.n.02').min_depth()
wn.synset('vertebrate.n.01').min_depth()
wn.synset('entity.n.01').min_depth()

# 路径相似度度量
right.path_similarity(minke)
right.path_similarity(orca)
right.path_similarity(tortoise)
right.path_similarity(novel)

# 文本语料库是一个大型结构化文本的集合
# 条件频率分布是频率分布的集合，每个分布都有不同的条件
# WordNet是一个面向语义的英文词典，由同义词的集合——或称为同义词集(synsets)——组成，并且组成一个网络
