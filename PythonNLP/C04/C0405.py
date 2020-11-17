# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0405.py
@Version    :   v0.1
@Time       :   2020-11-17 18:43
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？

# 4.5 更多函数操作(P164)
# 4.5.1 作为参数的函数
sent = ['Take', 'care', 'of', 'the', 'sense', ',',
        'and', 'the', 'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']


def extract_property(prop):
    return [prop(word) for word in sent]


# 将函数作为参数传入，并且在函数内调用
print("extract_property(len)= ", extract_property(len))


def last_letter(word):
    return word[-1]


# 将自定义函数作为参数传入，并且在函数内调用
print("extract_property(last_letter)= ", extract_property(last_letter))
# 将lambda表达式作为参数传入，并且在函数内调用
print("extract_property(lambda w: w[-1])= ", extract_property(lambda w: w[-1]))

# 将函数作为参数传给sorted()函数
# 这个版本的Python的sorted()函数不再支持函数传入
sorted(sent)


# 4.5.2 累积函数
# Ex4-5 累积输出到一个链表
def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result


# 全部搜索完毕再输出
print("search1:")
for i, item in enumerate(search1('zz', nltk.corpus.brown.words())):
    if i < 30:
        print(f"({i}){item}", end=', ')
        if i % 5 == 4:
            print()


# 生成器函数的累积输出
# 函数只产生调用程序需要的数据，并不需要分配额外的内存来存储输出
def search2(substring, words):
    for word in words:
        if substring in word:
            yield word


# 按照次序找到一个就输出一个
print("search2:")
i = 0
for i, item in enumerate(search2('zz', nltk.corpus.brown.words())):
    if i < 30:
        print(f"({i}){item}", end=', ')
        if i % 5 == 4:
            print()


# 更复杂的生成器的例子
def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]


print("1-permutation=", list(permutations(['police'])))
print("2-permutations=", list(permutations(['police', 'fish'])))
print("3-permutations=", list(permutations(['police', 'fish', 'buffalo'])))

# 4.5.3 高阶函数，函数式编程，Haskell
sent = ['Take', 'care', 'of', 'the', 'sense', ',',
        'and', 'the', 'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']


# 检查一个词是否来自一个开放的实词类
def is_content_word(word):
    return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']


# 将is_content_word()函数作为参数传入filter()函数中，就是高阶函数
print("高阶函数= ", list(filter(is_content_word, sent)))

# 下面是等价的代码
word_list = [
        w
        for w in sent
        if is_content_word(w)
]
print("word_list= ", word_list)

# 将len()函数作为参数传入 map()函数中，就是高阶函数
lengths = list(map(len, nltk.corpus.brown.sents(categories='news')))
sum(lengths) / len(lengths)  # 布朗语料库中句子的平均长度

# list()函数不支持将函数作为参数传入
# list(len, ['a', 'b', 'c'])

# 下面是等价的代码
lengths = [
        len(w)
        for w in nltk.corpus.brown.sents(categories='news')]
sum(lengths) / len(lengths)

# 多重函数嵌套
print(len(list(filter(lambda c: c.lower() in "aeiou", 'their'))))
print(list(map(lambda w: len(list(filter(lambda c: c.lower() in "aeiou", w))), sent)))

# 下面是等价的代码，以链表推导为基础的解决方案通常比基于高阶函数的解决方案可读性更好
print([
        len([
                c
                for c in w
                if c.lower() in "aeiou"
        ])
        for w in sent
])

print([
        [
                c
                for c in w
                if c.lower() in "aeiou"
        ]
        for w in sent
])

# 错误的输出结果
print([
        c
        for w in sent
        for c in w
        if c.lower() in "aeiou"
])


# 4.5.4 参数的命名(P167)
# 关键字参数：就是通过名字引用参数，
# 可以防止参数混淆，还可以指定默认值，还可以按做生意顺序访问参数
def repeat(msg='<empty>', num=1):
    return msg * num


print("repeat(num=3)= ", repeat(num=3))
print("repeat(msg='Alice')= ", repeat(msg='Alice'))
print("repeat(num=3, msg='Alice')= ", repeat(num=3, msg='Alice'))

dict = {'a': 1, 'b': 2, 'c': 3}
print("list(dict)= ", list(dict))
print("dict.keys()= ", dict.keys())
print("dict.values()= ", dict.values())
print("dict.items()= ", dict.items())
print("dict['a']= ", dict['a'])
print("dict.get('a')= ", dict.get('a'))


# 参数元组：*args 和 关键字参数字典：*kwargs
def generic(*args, **kwargs):
    for value in args:
        print('value={}'.format(value))
    print("kwargs['monty']= ", kwargs['monty'])
    for name in list(kwargs):
        print('name:value={}: {}'.format(name, kwargs[name]))


generic(1, "African swallow", monty='Python', lover='sy')

# zip() 函数对 *args 的支持
# *song 表示分解输入数据，等价于song[0],song[1],song[2],song[3]，分组的数目就是输入的数目
song = [['four', 'calling', 'birds'], ['three', 'French', 'hens'], ['two', 'turtle', 'doves'],
        ['one', 'tiger', 'animal']]
print("list(zip(song[0], song[1], song[2]))= ", list(zip(song[0], song[1], song[2])))
print("list(zip(*song))= ", list(zip(*song)))
print("list(zip(song))= ", list(zip(song)))


# 三种等效的方法调用函数 freq_words()
def freq_words(file, min=1, num=10):
    text = open(file, encoding='utf-8').read()
    tokens = nltk.word_tokenize(text)
    freqdist = nltk.FreqDist(t for t in tokens if len(t) >= min)
    return freqdist.most_common(num)


print("freq_words('func.py', 4, 10)= ", freq_words('func.py', 4, 10))
print("freq_words('func.py', min=4, num=10)= ", freq_words('func.py', min=4, num=10))
print("freq_words('func.py', num=10, min=4)= ", freq_words('func.py', num=10, min=4))


# 增加了 verbose 参数，可以设置为调试开关
def freq_words(file, min=1, num=10, verbose=False):
    freqdist = nltk.FreqDist()
    if verbose:
        print('Opening', file)
    with open(file, encoding='utf-8') as f:
        text = f.read()
    if verbose:
        print('Read in %d characters' % len(file))
    for word in nltk.word_tokenize(text):
        if len(word) >= min:
            freqdist[word] += 1
            if verbose and freqdist.N() % 100 != 0:
                print('.', end='')
    if verbose:
        print()
    return freqdist.most_common(num)


freq_words('func.py', 4, 10, True)
