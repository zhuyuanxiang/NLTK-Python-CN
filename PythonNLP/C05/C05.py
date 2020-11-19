# Ch5 分类和标注词汇
# 词性标注（parts-of-speech tagging，POS tagging）：简称标注。
# 将词汇按照它们的词性（parts-of-speech，POS）进行分类并对它们进行标注
# 词性：也称为词类或者词汇范畴。
# 用于特定任务标记的集合被称为一个标记集。

import nltk
import pylab
from nltk import word_tokenize
from nltk.corpus import brown

brown_words = brown.words(categories='news')
brown_tagged_words = brown.tagged_words(categories='news')
brown_sents = brown.sents(categories='news')
brown_tagged_sents = brown.tagged_sents(categories='news')

# Sec 5.1 使用词性标注器
text = word_tokenize("And now for something completely different")
nltk.pos_tag(text)
nltk.help.upenn_tagset('CC')
nltk.help.upenn_tagset('RB')
nltk.help.upenn_tagset('IN')
nltk.help.upenn_tagset('NN')
nltk.help.upenn_tagset('JJ')
nltk.corpus.brown.readme()
print(nltk.corpus.gutenberg.readme())

# 处理同形同音异义词，系统正确标注了
# 前面的refUSE是动词，后面的REFuse是名词
# 前面的permit是动词，后面的permit是名字
text = word_tokenize("They refuse to permit us to obtain the refuse permit")
text = word_tokenize("They refuse to permit us to obtain the beautiful book")
nltk.pos_tag(text)

# 找出形如w1 w w2的上下文，然后再找出所有出现在相同上下文的词 w'，即w1 w' w2
# 用于寻找相似的单词，因为这些单词处于相同的上下文中
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('word')
text.similar('woman')
text.similar('bought')
text.similar('over')
text.similar('the')

# Sec 5.2 标注语料库
# str2tuple() 将已经标注的字符串转换成元组
taggen_token = nltk.tag.str2tuple('fly/NN')
taggen_token
taggen_token[0]
taggen_token[1]

sent = '''
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.
'''

sent.split()
[nltk.tag.str2tuple(t) for t in sent.split()]

# 读取已经标注的语料库
# 打开brown语料库的ca01文件，可以看到下面的内容：
# The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at
# investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd
# ``/`` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./.
# 这个是已经标注好的语料库，可以使用函数tagged_words()直接读取

nltk.corpus.brown.tagged_words()
nltk.corpus.brown.tagged_words(tagset='universal')  # 使用通用标注集进行词类标注

nltk.corpus.treebank.tagged_words()
nltk.corpus.treebank.tagged_words(tagset='universal')

nltk.corpus.nps_chat.tagged_words()
nltk.corpus.nps_chat.tagged_words(tagset='universal')

nltk.corpus.conll2000.tagged_words()
nltk.corpus.conll2000.tagged_words(tagset='universal')

# ToDo: 以下的都无法正常转换为通用标注集
nltk.corpus.sinica_treebank.tagged_words()
nltk.corpus.sinica_treebank.tagged_words(tagset='universal')

nltk.corpus.indian.tagged_words()
nltk.corpus.indian.tagged_words(tagset='universal')

nltk.corpus.mac_morpho.tagged_words()
nltk.corpus.mac_morpho.tagged_words(tagset='universal')

nltk.corpus.cess_cat.tagged_words()
nltk.corpus.cess_cat.tagged_words(tagset='universal')

# 使用tagged_sents()可以直接把语料库分割成句子，而不是将所有的词表示成一个链表，句子中的词同样进行了词类标注。
# 因为开发的自动标注器需要在句子链表上进行训练和测试，而不是在词链表上。
nltk.corpus.brown.tagged_sents()[0]
nltk.corpus.brown.tagged_sents(tagset='universal')[0]

# 2.3 A Universal Part-of-Speech Tagset, 一个通用的（简化的）标注集
# http://www.nltk.org/book/ch05.html Table2.1 （比书P200 表5-1还要简单）
# Tag 	Meaning 	            English Examples
# ADJ 	adjective 	            new, good, high, special, big, local
# ADP 	adposition 	            on, of, at, with, by, into, under
# ADV 	adverb 	                really, already, still, early, now
# CONJ 	conjunction 	        and, or, but, if, while, although
# DET 	determiner, article 	the, a, some, most, every, no, which
# NOUN 	noun 	                year, home, costs, time, Africa
# NUM 	numeral 	            twenty-four, fourth, 1991, 14:24
# PRT 	particle 	            at, on, out, over per, that, up, with
# PRON 	pronoun 	            he, their, her, its, my, I, us
# VERB 	verb 	                is, say, told, given, playing, would
# . 	punctuation marks 	    . , ; !
# X 	other 	                ersatz, esprit, dunno, gr8, univeristy

from nltk.corpus import brown

brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
print(list(tag_fd))
tag_fd.keys()
tag_fd.most_common()
tag_fd.tabulate()
tag_fd.plot(cumulative=True)

# 图形化的POS一致性工具，可以用来寻找任一词和POS标记的组合
# 例如："VERB VERB" 或者 "was missing" 或者 "had VERB" 或者 "DET money" 等等
nltk.app.concordance()

# 2.4 名词：一般指人、地点、事情和概念。可能出现在限定词和形容词之后，可以是动词的主语或者宾语。
# 表5-2 名词的句法模式
# 统计构成二元模型（W1，W2）中W2=‘NOUN’的W1的词性的比例
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
word_tag_pairs = nltk.bigrams(brown_news_tagged)  # 构建双词链表
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
fdist = nltk.FreqDist(noun_preceders)
fdist.most_common()
[tag for (tag, _) in fdist.most_common()]
fdist.plot(cumulative=True)
# 结论：名词出现在限定词和形容词之后，包括数字形容词（即数词，标注为NUM）

# 2.5 动词：描述事件和行动的词。在句子中，动词通常表示涉及一个或多个名词短语所指示物的关系。
# 表5-3 动词的句法模式
# 找出新闻文本中最常见的动词（频率分布中计算的项目是词——标记对）
# wsj = nltk.corpus.treebank.tagged_words(simplify_tags=True)   # simplify_tags 不再支持
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
word_tag_fd.most_common(20)
word_tag_fd.tabulate()
[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB'][:20]
list(word_tag_fd)[:20]
fdist = nltk.FreqDist(word_tag_fd)
fdist.most_common(12)
fdist.tabulate()
# fdist.plot(cumulative=True)   # 不能执行，会死机，因为动词单词数目太多

wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_pairs = nltk.bigrams(wsj)
verb_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'VERB']
fdist = nltk.FreqDist(verb_preceders)
fdist.most_common()
fdist.tabulate()
fdist.plot(cumulative=True)
# 结论：动词出现在动词、名字和副词后面。

# 因为词汇和标记是成对的，所以把词汇作为条件，把标记作为事件，使用条件——事件对的链表初始化条件频率分布。
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].most_common(20)
cfd1['cut'].most_common(20)
cfd1.tabulate()
list(cfd1)[:20]

# 也可以颠倒配对，把标记作为条件，词汇作为事件，生成条件频率分布，就可以直接查找标记对应哪些词了。
wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
cfd2['VBN']
cfd2['VBN'].keys()
list(cfd2['VBN'])[:20]
list(cfd2['VBN'].keys())[:20]
cfd2['VBN'].most_common(20)
'been' in cfd2['VBN'].keys()

# 尝试分辨VD（过去式）和VN（过去分词）之间的区别
# 先找出同是VD和VN的词汇，然后分析它们的上下文区别
wsj = nltk.corpus.treebank.tagged_words()
cfd3 = nltk.ConditionalFreqDist(wsj)
# cfd1.conditions() 返回所有的条件构成的链表，等价于list(cfd1.keys())返回所有的关键字。
[w for w in cfd3.conditions() if 'VBD' in cfd3[w] and 'VBN' in cfd3[w]]
cfd3['kicked']
idx1 = wsj.index(('kicked', 'VBD'))
idx2 = wsj.index(('kicked', 'VBN'))
' '.join(word for word, tag in wsj[idx1 - 10:idx1 + 10])
' '.join(word for word, tag in wsj[idx2 - 10:idx2 + 10])


# 其他词类（形容词、副词、介词、冠词（限定词）、情态动词、人称代词）
# 形容词：修饰名词，可以作为修饰符 或 谓语。
# 副词：修饰动词，指定时间、方式、地点或动词描述的事件发展方向；修饰形容词。

# P204 2.7 未简化的标记
# Ex5-1 找出最频繁的名词标记的程序
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())


tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
    print(tag, tagdict[tag])

# 2.8. 探索已经标注的语料库
# 观察 often 后面的词汇
brown_learned_text = nltk.corpus.brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))

brown_learned_tagged = nltk.corpus.brown.tagged_words(categories='learned', tagset='universal')
brown_learned_bigrams = nltk.bigrams(brown_learned_tagged)
[(a, b) for (a, b) in brown_learned_bigrams]

brown_learned_bigrams = nltk.bigrams(brown_learned_tagged)
list(brown_learned_bigrams)

brown_learned_bigrams = nltk.bigrams(brown_learned_tagged)
tags = [b[1] for (a, b) in nltk.bigrams(brown_learned_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()
fd.plot(cumulative=True)

# P205 Ex5-2 使用POS标记寻找三词短语(<Verb>to<Verb>)
from nltk.corpus import brown


def process(sentence):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):
        if t1.startswith('V') and t2 == 'TO' and t3.startswith('V'):
            print(w1, w2, w3)


for tagged_sent in nltk.corpus.brown.tagged_sents():
    if len(tagged_sent) >= 3:
        process(tagged_sent)

brown_news_tagged = brown.tagged_words(categories='news')
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)
for word in sorted(data.conditions()):
    if len(data[word]) > 3:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, ' '.join(tags))

data['works']
data['$1']
data['$222']
data.tabulate()
data.conditions()
data.values()

nltk.app.concordance()

# P206 3 使用Python字典映射词及其属性
# Python字典数据类型（以称为关联数组或者哈希数组），学习如何使用字典表示包括词性在内的各种不同语言信息
# 3.1 索引链表 与 字典 的区别

# 3.2. Python字典
pos = {}
pos['colorless'] = 'ADJ'
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
pos['colorless']
pos

# 访问不存在的键
pos['green']

# 字典转换成链表
list(pos)
# 字典排序
sorted(pos)
# 字典顺序访问
[w for w in pos if w.endswith('s')]

for word in sorted(pos):
    print(word + ":", pos[word])

pos.keys()  # 键
pos.values()  # 值
pos.items()  # 对

for key, val in sorted(pos.items()):
    print(key + ":", val)

# 字典中键必须惟一
pos['sleep'] = 'V'
pos['sleep']
pos['sleep'] = 'N'
pos['sleep']

# 3.3. 定义字典（创建字典的两种方式）
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos = dict(colorless='ADJ', ideas='N', sleep='V', furiously='ADV')

# 3.4. 默认字典（字典创建新键时的默认值）
from collections import defaultdict

frequency = defaultdict(int)  # 默认值可以是不变对象
frequency['colorless'] = 4
frequency['colorless']
frequency['ideas']  # 访问不存在的键时，自动创建，使用定义的默认值
list(frequency.items())

pos = defaultdict(list)  # 默认值也可以是可变对象
pos['sleep'] = ['NOUN', 'VERB']
pos['sleep']
pos['ideas']
list(pos.items())


class myObject():
    def __init__(self, data=0):
        self._data = data
        return


oneObject = myObject(5)
oneObject._data
twoObject = myObject()
twoObject._data

pos = defaultdict(myObject)
pos['sleep'] = myObject(5)
pos['ideas']
list(pos.items())
pos['sleep']._data
pos['ideas']._data

# 默认 lambda 表达式
pos = defaultdict(lambda: 'NOUN')
pos['colorless'] = 'ADJ'
pos['colorless']
pos['blog']
list(pos.items())

# 使用 UNK(out of vocabulary)（超出词汇表）标识符来替换低频词汇
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = [word for (word, _) in vocab.most_common(1000)]
mapping = defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
list(mapping.items())
alice2 = [mapping[v] for v in alice]
alice2[:100]

# 3.5. 递增地更新字典
# Ex5-3 递增地更新字典，按值排序
counts = nltk.defaultdict(int)
for (word, tag) in nltk.corpus.brown.tagged_words(categories='news', tagset='universal'):
    counts[tag] += 1
counts['NOUN']
sorted(counts)
counts

from operator import itemgetter

sorted(counts.items(), key=itemgetter(0), reverse=False)
sorted(counts.items(), key=itemgetter(1), reverse=False)
sorted(counts.items(), key=itemgetter(1), reverse=True)
sorted(counts.items(), key=itemgetter(2), reverse=False)  # IndexError: tuple index out of range
[t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)]

pair = ('NP', 8336)
pair
pair[1]
itemgetter(1)(pair)
itemgetter(0)(pair)

# 通过最后两个字母索引词汇
last_letters = defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)

last_letters['ly']
last_letters['xy']

# 颠倒字母而成的字（回文构词法，相同字母异序词，易位构词，变位词）索引词汇
anagrams = defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)
anagrams['aeilnrt']
anagrams['kloo']
anagrams['Zahity']
anagrams[''.join(sorted('love'))]

# NLTK 提供的创建 defaultdict(list) 更加简便的方法
# nltk.Index() 是对 defaultdict(list) 的支持
# nltk.FreqDist() 是对 defaultdict(int) 的支持（附带了排序和绘图的功能）
anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
anagrams['aeilnrt']

anagrams = nltk.FreqDist(''.join(sorted(w)) for w in words)
anagrams.most_common(20)

# 3.6. 复杂的键和值
# 使用复杂的键和值的默认字典
pos = defaultdict(lambda: defaultdict(int))
brown_news_tagged = nltk.corpus.brown.tagged_words(categories='news', tagset='universal')
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1

pos[('DET', 'right')]
pos[('NOUN', 'further')]
pos[('PRT', 'book')]

# 3.7. 颠倒字典
# 通过键查值速度很快，但是通过值查键的速度较慢，为也加速查找可以重新创建一个映射值到键的字典
counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word] += 1

# 通过值查键的一种方法
[key for (key, value) in counts.items() if value == 32]

# pos 是键-值对字典；pos2 是值-键对字典
pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos2 = dict((value, key) for (key, value) in pos.items())
pos2['N']

# 一个键有多个值就不能使用上面的重建字典的方法，下面提供了一个新的创建值-键对字典的方法
pos.update({'cats': 'N', 'scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
pos2 = defaultdict(list)
for key, value in pos.items():
    pos2[value].append(key)

pos2['ADV']

# 使用 nltk.Index() 函数创建新的值-键对字典
pos2 = nltk.Index((value, key) for (key, value) in pos.items())
pos2['ADV']

# 4. 自动标注（利用不同的方式给文本自动添加词性标记）
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
brown_tagged_words = brown.tagged_words(categories='news')
brown_words = brown.words(categories='news')

# 4.1. 默认标注器
# 寻找在布朗语料库中新闻类文本使用次数最多的标记
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()

# 因为 'NN' 是使用次数最多的标记，因此设置它为默认标注
raw = 'I do not lie green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
default_tagger.evaluate(brown_tagged_sents)  # 评测默认标注的正确率

# 4.2. 正则表达式标注器
patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'(a|an)', 'AT'),
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]

regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])  # 是标注的文本
regexp_tagger.evaluate(brown_tagged_sents)  # brown_tagged_sents 是测试集

# 4.3. 查询标注器
# 找出100最频繁的词，存储它们最有可能的标记，然后使用这个信息作为“查找标注器”的模型
fd = nltk.FreqDist(brown_words)
cfd = nltk.ConditionalFreqDist(brown_tagged_words)
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
cfd['the']

# 一元语法模型，统计词料库中每个单词标注最多的词性作为一元语法模型的建立基础
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

sent = brown_sents[3]
baseline_tagger.tag(sent)

# 对于一元语法模型不能标注的单词，使用默认标注器，这个过程叫做“回退”。
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
baseline_tagger.evaluate(brown_tagged_sents)


# Ex5-4 查找标注器的性能评估
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown_tagged_sents)


def display():
    word_freqs = nltk.FreqDist(brown_words).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown_tagged_words)
    sizes = 2 ** pylab.arange(15)
    # 单词模型容量的大小对性能的影响
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()


display()

# 5 N元语法标注器
# xxxTagger() 只能使用 sent 作为训练语料
# 5.1 一元标注器，统计词料库中每个单词标注最多的词性作为一元语法模型的建立基础
# 使用训练数据来评估一元标注器的准确度
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
unigram_tagger.evaluate(brown_tagged_sents)

# 5.2 将数据分为 训练集 和 测试集
# 使用训练数据来训练一元标注器，使用测试数据来评估一元标注器的准确度
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

# 5.3 更加一般的N元标注器
# 二元标注器
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])  # 标注训练集中数据
bigram_tagger.tag(brown_sents[4203])  # 标注测试集中数据
bigram_tagger.evaluate(test_sents)  # 整体准确度很低，是因为数据稀疏问题

# 5.4 组合标注器，效果更差，为什么？
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t1.evaluate(test_sents)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)  # 这个效果最好
t3 = nltk.TrigramTagger(train_sents, backoff=t2)
t3.evaluate(test_sents)

t2 = nltk.BigramTagger(train_sents, cutoff=1, backoff=t1)
t2.evaluate(test_sents)
# cutoff=15时，准确率高，可见上下文并不能真正提示单词标注的内在规律
t3 = nltk.TrigramTagger(train_sents, cutoff=15, backoff=t2)
t3.evaluate(test_sents)

# 5.5 标注未知的单词
# 对于生词。可以使用回退到正则表达式标注器或者默认标注器，但是都无法利用上下文。

# 5.6 标注器的存储
from pickle import dump, load

output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

input = open('t2.pkl', 'rb')
t2_bak = load(input)
input.close()

text = """The board's action shows what free enterprise
    is up against in our complex maze of regulatory laws ."""
tokens = text.split()
t2.tag(tokens)
t2_bak.tag(tokens)
t2.evaluate(test_sents)
t2_bak.evaluate(test_sents)

# 5.7. N元标注器的性能边界（上限）
# 一种方法是寻找有歧义的单词的数目，大约有1/20的单词可能有歧义
# cfd无法正确赋值，因为有些句子的长度少于3个单词，影响了trigrams()函数的正确运行
cfd = nltk.ConditionalFreqDist(
    ((x[1], y[1], z[0]), z[1])
    for sent in brown_tagged_sents if len(sent)>=3
    for x, y, z in nltk.trigrams(sent) )
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()

# Colquitt 就是那个错误的句子，在ca01文本文件中可以找到
for sent in brown_tagged_sents[:100]:
    print(sent,len(sent))
    if len(sent)>=3:
        for x, y, z in nltk.trigrams(sent):
            print(x[0], y[0], z[0], x[1], y[1], z[1])

# 一种方法是研究被错误标记的单词
# ToDo: 可是显示出来的结果根本没有可视性呀？
test_tags = [tag for sent in brown.sents(categories='editorial') for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print(nltk.ConfusionMatrix(gold_tags, test_tags))

# 跨句子边界的标注
# 使用三元标注器时，跨句子边界的标注会使用上个句子的最后一个词+标点符号+这个句子的头一个词
# 但是，两个句子中的词并没有相关性，因此需要使用已经标注句子的链表来训练、运行和评估标注器
# Ex5-5 句子层面的N-gram标注
# 前面的组合标注器已经是跨句子边界的标注

# 6.基于转换的标注
# n-gram标注器存在的问题：
# 1）表的大小（语言模型），对于trigram表会产生巨大的稀疏矩阵
# 2）上下文。n-gram标注器从上下文中获得的唯一信息是标记，而忽略了词本身。
# 在本节中，利用Brill标注，这是一种归纳标注方法，性能好，使用的模型仅有n-gram标注器的很小一部分。
# Brill标注是基于转换的学习，即猜想每个词的标记，然后返回和修正错误的标记，陆续完成整个文档的修正。
# 与n-gram标注一样，需要监督整个过程，但是不计数观察结果，只编制一个转换修正规则链表。
# Brill标注依赖的原则：规则是语言学可解释的。因此Brill标注可以从数据中学习规则，并且也只记录规则。
# 而n-gram只是隐式的记住了规律，并没有将规律抽象出规则，从而记录了巨大的数据表。
# Brill转换规则的模板：在上下文中，替换T1为T2.
# 每一条规则都根据其净收益打分 = 修正不正确标记的数目 - 错误修改正确标记的数目
from nltk.tbl import demo as brill_demo

brill_demo.demo()
print(open('errors.out').read())

# 7. 如何确定一个词的分类（词类标注）
# 语言学家使用形态学、句法、语义来确定一个词的类别

# 7.1. 形态学线索：词的内部结构有助于词类标注。

# 7.2. 句法线索：词可能出现的典型的上下文语境。

# 7.3. 语义线索：词的意思

# 7.4. 新词（未知词）的标注：开放类和封闭类

# 7.5. 词性标记集中的形态学
# 普通标记集捕捉的构词信息：词借助于句法角色获得的形态标记信息。
# 大多数词性标注集都使用相同的基本类别。更精细的标记集中包含更多有关这些形式的信息。
# 没有一个“正确的方式”来分配标记，只能根据目标不同而产生的或多或少有用的方法

# 8. 小结
# 词可以组成类，这些类称为词汇范畴或者词性。
    # 词性可以被分配短标签或者标记
# 词性标注、POS标注或者标注：给文本中的词自动分配词性的过程
# 语言词料库已经完成了词性标注
# 标注器可以使用已经标注过的语料库进行训练和评估
# 组合标注方法：把多种标注方法（默认标注器、正则表达式标注器、Unigram标注器、N-gram标注器）利用回退技术结合在一起使用
# 回退是一个组合模型的方法：当一个较为专业的模型不能为给定内容分配标记时，可以回退到一个较为一般的模型
# 词性标注是序列分类任务，通过利用局部上下文语境中的词和标记对序列中任意一点的分类决策
# 字典用来映射任意类型之间的信息
# N-gram标注器可以定义为不同数值的n，当n过大时会面临数据稀疏问题，即使使用大量的训练数据，也只能够看到上下文中的一部分
# 基于转换的标注包括学习一系列的“改变标记s为标记t在上下文c中”形式的修复规则，每个规则都可以修复错误，但是也可能会引入新的错误
