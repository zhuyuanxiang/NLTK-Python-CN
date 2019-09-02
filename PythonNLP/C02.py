import nltk

# Chap2 获取文本语料库 和 词汇资源
# 1. 获取文本语料库
# 1.1. 古腾堡语料库
nltk.corpus.gutenberg.fileids()
emma_words = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma_words)
emma_text = nltk.Text(emma_words)
emma_text.concordance('surprise')

from nltk.corpus import gutenberg

gutenberg.fileids()
emma_words = gutenberg.words('austen-emma.txt')

# 平均词长、平均句子长度、文中每个单词出现的平均次数（词汇多样性得分）。文件名称
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars / num_words), round(num_words / num_sents), round(num_words / num_vocab), fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences[1116]
longest_len = max(len(s) for s in macbeth_sentences)
longest_sent = [s for s in macbeth_sentences if len(s) == longest_len]
' '.join(longest_sent[0])

# 1.2. 网络文本 和 聊天文本
from nltk.corpus import webtext

for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

from nltk.corpus import nps_chat

chatroom = nps_chat.posts('10-19-20s_706posts.xml')
' '.join(chatroom[123])

for fileid in nps_chat.fileids():
    print(fileid, ' '.join(nps_chat.posts(fileid)[123]))

# 1.3. Brown（布朗）语料库：用于研究文体之间的系统性差异（又叫文体学研究）
from nltk.corpus import brown

brown.categories()
brown_news_words = brown.words(categories='news')
brown_cg22_words = brown.words(fileids='cg22')
brown_sents = brown.sents(categories=['news', 'editorial', 'reviews'])

fdist = nltk.FreqDist([w.lower() for w in brown_news_words])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=' ')

cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories() for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

# 1.4. 路透社语料库
from nltk.corpus import reuters

reuters.fileids()
reuters.categories()
reuters.categories('training/9865')
reuters.categories('test/14826')
reuters.categories(['training/9865', 'test/14826'])
reuters.fileids('barley')
reuters.fileids('corn')
reuters.fileids(['barley', 'corn'])

len(reuters.words('training/9865'))
len(reuters.words('test/14826'))
len(reuters.words(['training/9865', 'test/14826']))
len(reuters.words(categories='barley'))
len(reuters.words(categories='corn'))
len(reuters.words(categories=['barley', 'corn']))

# 1.5. 美国总统就职演说语料库
from nltk.corpus import inaugural

inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

# 统计演说词中 america 和 citizen 出现的次数
cfd = nltk.ConditionalFreqDist((target, fileid[:4])
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen']
                               if w.lower().startswith(target))
cfd.plot()

# 1.6. 标注文本语料库：表2-2 NLTK中的一些语料库和语料库样本
# 1.7. 其他语言的语料库
# 在Python 3.0中已经不存在字符编码问题
cess_esp_words = nltk.corpus.cess_esp.words()
print(cess_esp_words[:35])

floresta_words = nltk.corpus.floresta.words()
print(floresta_words[:35])

indian_words = nltk.corpus.indian.words()
print(indian_words[:35])

udhr_fileids = nltk.corpus.udhr.fileids()
print(udhr_fileids[:35])

udhr_words = nltk.corpus.udhr.words('Javanese-Latin1')
print(udhr_words[:35])

from nltk.corpus import udhr

languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)

languages = ['Chickasaw-Latin1', 'English-Latin1', 'German_Deutsch-Latin1', 'Greenlandic_Inuktikut-Latin1',
             'Hungarian_Magyar-Latin1', 'Ibibio_Efik-Latin1']  # , 'Chinese_Mandarin-UTF8']
cfd = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in udhr.words(lang))
cfd.plot(cumulative=True)
cfd.tabulate(samples=range(10), cumulative=True)
cfd.tabulate(conditions=['English-Latin1', 'German_Deutsch-Latin1'], samples=range(10), cumulative=True)
# 中文是字符型的，不能使用单词读入
chinese_mandarin_raw = udhr.raw('Chinese_Mandarin-UTF8')
chinese_mandarin_words = udhr.words('Chinese_Mandarin-UTF8')
chinese_mandarin_sents = udhr.sents('Chinese_Mandarin-UTF8')


def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()


# 1.8. 文本语料库的结构
raw = gutenberg.raw('burgess-busterbrown.txt')
raw[1:20]
words = gutenberg.words('burgess-busterbrown.txt')
words[1:20]
sents = gutenberg.sents('burgess-busterbrown.txt')
sents[1:20]

# 1.9. 载入自己的语料库
from nltk.corpus import PlaintextCorpusReader

corpus_root = '/Temp/delete'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('blake-poems.txt')

from nltk.corpus import BracketParseCorpusReader

corpus_root = r'C:\nltk_data\corpora\treebank\combined'
file_pattern = r'.*/wsj_.*\.mrg'
file_pattern = r'wsj_.*.mrg'
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()
len(ptb.sents())
ptb.sents(fileids='wsj_0199.mrg')[1]

# 2. 条件频率分布：是频率分布的集合，每个频率分布有一个不同的“条件”。(condition,word)根据condition（条件）统计word（单词）的频率。
# 2.1. 条件 和 事件
# 2.2. 按文体计数词汇
from nltk.corpus import brown

cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories() for word in brown.words(categories=genre))
genre_word = [(genre, word) for genre in ['news', 'romance'] for word in brown.words(categories=genre)]
len(genre_word)
genre_word[:4]
genre_word[-4:]
cfd = nltk.ConditionalFreqDist(genre_word)
cfd
cfd.conditions()
print(cfd['news'])
print(cfd['romance'])
cfd['romance'].most_common(20)
cfd['romance']['could']

# 2.3. 绘制分布图 显示分布表
from nltk.corpus import inaugural

cfd = nltk.ConditionalFreqDist((target, fileid[:4])
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen']
                               if w.lower().startswith(target))
from nltk.corpus import udhr

languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word)) for lang in languages for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
cfd.tabulate()
cfd.tabulate(cumulative=True)
cfd.tabulate(samples=range(10), cumulative=True)
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)

from nltk.corpus import brown

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
brown_cfd = nltk.ConditionalFreqDist((genre, word)
                                     for genre in ['news', 'romance']
                                     for word in brown.words(categories=genre)
                                     if word in days)
# ToDo: 控制星期输出的顺序？

# 2.4. 使用二元语法（双边词）生成随机文本
# nltk.bigrams()生成连续的词对链表
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.for ']
list(nltk.bigrams(sent))


# P59 Ex2-1 产生随机文本
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()  # 选择与原单词匹配度最大的单词作为下一个单词


text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

# 打印出来的结果 和 直接访问的结果不一样
print(cfd['living'])
cfd['living']
cfd['living'].max()
cfd['finding'].max()

generate_model(cfd, 'living')
generate_model(cfd, 'beginning')
generate_model(cfd, 'finding')

# 3. Python 中的 代码重用
# 3.1. 使用文本编辑器创建Python程序，还是用PyCharm吧
# 3.2. 如何定义函数
# 导入精确除法，现在默认是精确除法，使用截断除法需要使用 3//4
from __future__ import division


def lexical_diversity(text):
    return len(text) / len(set(text))


def lexical_diversity(my_text_data):
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = vocab_size / word_count
    return diversity_score


# 3.3 Python 模块
from PythonNLP.textproc import plural

# 修改 textproc 中的代码时，需要重启 Python Console 环境
plural('wish')
plural('fan')


# 4. 词典资源
# 4.1. 词汇列表语料库
# 过滤文本，删除掉常见英语词典中的单词，留下罕见的或者拼写错误的词汇
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)


unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
unusual_words(nltk.corpus.nps_chat.words())

# 下面这个是停止词语料库（不是不再使用的单词，是会使说话停止的单词）
from nltk.corpus import stopwords

stopwords.words('english')


def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


content_fraction(nltk.corpus.reuters.words())
# 利用停止词，筛选掉文本中三分之一的单词

# 使用 ‘egivrvonl’ 字母可以组成多少个不少于4个字母的单词？
puzzle_letters = nltk.FreqDist('egivrvonl')
puzzle_letters.tabulate()
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters]

# 使用名字语料库
names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]
len(male_names)
len(female_names)
len(set(male_names).difference(female_names))
set([1, 2, 3, 4]).difference(set([4, 5, 6, 7]))

from nltk.corpus import names

names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

name_ends = ((fileid, name[-2:]) for fileid in names.fileids() for name in names.words(fileid))
for name_end in name_ends:
    print(name_end)
cfd = nltk.ConditionalFreqDist((fileid, name[-2:]) for fileid in names.fileids() for name in names.words(fileid))
cfd.tabulate()
cfd.plot()  # 图2-7 显示男性与女性名字的结尾字母

# 4.2. 发音词典
entries = nltk.corpus.cmudict.entries()
len(entries)
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
[word for word, pron in entries if pron[-4:] == syllable]

# 寻找拼写以'n'结尾，发音以'M'结尾的单词
[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']


# 定义提取重音数字的函数
def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]


[(w, pron) for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]

# 拆分映射、列表、字符串的测试
ex_pron = ('surrealistic', ['S', 'ER0', 'IY2', 'AH0', 'L', 'IH1', 'S', 'T', 'IH0', 'K'])
(word, pron) = ex_pron
for phone in pron:
    print(phone)
    for char in phone:
        print(char)

# P69 使用条件频率分布寻找相应词汇的最小对比集，找到所有p开头的三音素词，并比照它们的第一个和最后一个音素来分组
p3 = [(pron[0] + '-' + pron[2], word) for (word, pron) in entries if pron[0] == 'P' and len(pron) == 3]

cfd = nltk.ConditionalFreqDist(p3)
cfd.tabulate(conditions=['P-P', 'P-R'])
cfd['P-P']
for template in cfd.conditions():
    if len(cfd[template]) > 10:
        words = cfd[template].keys()
        wordlist = ' '.join(words)
        print(template, wordlist[:70] + "...")

# 访问词典的方式
prondict = nltk.corpus.cmudict.dict()
prondict['fire']
prondict['blog']  # 词典中没有，报错KeyError
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
prondict['blog']

# 在词典中寻找单词的发音
text = ['natural', 'language', 'processing']
[ph for w in text for ph in prondict[w][0]]

# 加[0]是因为natural有两个发音，取其中一个就好了
[ph for w in text for ph in prondict[w]]
prondict['natural']

# P70 4.3. 比较词表（Swadesh wordlists），包括几种语言的约200个常用词的列表，可以用于比较两个语言之间的差别，也可以用于不同语言的单词翻译
from nltk.corpus import swadesh

swadesh.fileids()
swadesh.words('en')

fr2en = swadesh.entries(['fr', 'en'])
fr2en
translate = dict(fr2en)
translate['chien']

de2en = swadesh.entries(['de', 'en'])
translate.update(dict(de2en))
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(es2en))
translate['jeter']
translate['Hund']
translate['perro']

languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])

# P71 4.4. 词汇工具 Toolbox Shoebox，是由一些条目的集合组成，每个条目由一个或多个字段组成，大多数字段都是可选的或者重复的

from nltk.corpus import toolbox

rotokas = toolbox.entries('rotokas.dic')
for word in rotokas:
    print(word)

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
