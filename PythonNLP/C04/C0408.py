from tools import *

# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？
# 4.8 Python库的样例(P183)
# 4.8.1 Matplotlib 绘图工具

# Ex4-10：布朗语料库中不同部分的情态动词频率
colors = 'rgbcmyk'  # red, green, blue, cyan, magenta, yellow, black


def bar_chart(categories, words, counts):
    """Plot a bar chart showing counts for each word by category"""
    import pylab
    ind = pylab.arange(len(words))
    width = 1 / (len(categories) + 1)
    bar_groups = []
    for c in range(len(categories)):
        bars = plt.bar(ind + c * width, counts[categories[c]], width, color=colors[c % len(colors)])
        bar_groups.append(bars)
    plt.xticks(ind + width, words)
    plt.legend([b[0] for b in bar_groups], categories, loc='upper left')
    plt.ylabel('Frequency')
    plt.title('Frequency of Six Modal Verbs by Genre')
    plt.show()


genres = ['news', 'religion', 'hobbies', 'government', 'adventure']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfdist = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in genres
        for word in nltk.corpus.brown.words(categories=genre)
        if word in modals
)

counts = {}
for genre in genres:
    counts[genre] = [cfdist[genre][word] for word in modals]

bar_chart(genres, modals, counts)

import sys

sys.setrecursionlimit(100000)  # 设置递归深度为100000，避免递归深度不够，导致pyplot.savefig()报错

from matplotlib import use, pyplot

use('Agg')
pyplot.savefig('modals.png')  # RecursionError: maximum recursion depth exceeded，因为递归深度超过1000
print('Content-Type: text/html')
print()
print('<html><body>')
print('<img src="modals.png"/>')
print('</body></html>')

# 4.8.2 NetworkX(P182)
# NetworkX包定义和操作由节点和边组成的结构（称为图）。

# Ex4-11：使用NetworkX 和 Matplotlib 结合来可视化WordNet的网络结构（语义网络）
import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn


def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)


def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G


def graph_draw(graph):
    nx.draw(graph,
            node_size=[16 * graph.degree(n) for n in graph],
            node_color=[graph.depth[n] for n in graph],
            with_labels=False)
    matplotlib.pyplot.show()


dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)

# 4.8.3 CSV(Comma-separated values, 逗号分隔型取值格式)
import csv

input_file = open('lexicon.csv', 'rb')  # 不能使用'b'，即bytes格式读取会报错
input_file = open('lexicon.csv')
print(list(csv.reader(input_file)))
input_file.seek(0)  # 文件内容读出后，需要重新将指针归零，才能再次读出数据
for row in csv.reader(input_file):
    print(row)

# 4.8.4 NumPy
# NumPy 提供了多维数组对象
from numpy import array

cube = array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
              [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
              [[6, 6, 6], [7, 7, 7], [8, 8, 8]]])
print("cube[1, 1, 1]= ", cube[1, 1, 1])
show_subtitle("cube[2, 1:]")
print(cube[2, 1:])
show_subtitle("cube[2].transpose()")
print(cube[2].transpose())

# Numpy 提供了线性代数函数。
# 可以进行矩阵的奇异值分解，应用在潜在语义分析中，帮助识别文档集合中的隐含概念。
from numpy import linalg

a = array([[4, 0], [3, -5]])
u, s, vt = linalg.svd(a)
print(u)
print(s)
print(vt)

# 4.8.5 其他 Python 库
# -   关系数据库：mysql-python
# -   大数据集合：PyLucene
# -   PDF: pypdf
# -   MSWord: pywin32
# -   XML: xml.etree
# -   RSS: feedparser
# -   e-mail: imaplib, email

# 4.9 小结
# -   使用对象引用进行 Python 赋值和参数传递。
# -   使用 is 测试对象是否相同，使用 == 测试对象是否相等
# -   字符串、链表 和 元组 是不同类型的序列对象，支持常用的序列操作(索引、切片、len()、sorted()、使用 in 的成员测试)
# -   通过打开文件将文本写入到文件中
# -   声明式的编程风格使用代码更加简洁可读
# -   函数是编程的抽象化过程，关键概念(参数传递、变量范围 和 docstrings)
# -   函数作为命名空间，在函数内部定义的变量名称在函数外部是不可见的，除非宣布为全局变量
# -   模块允许将材料与本地的文件逻辑关联起来。模块作为命名空间，在模块内部定义的变量和函数在模块外部是不可见的，除非这些名称被其他模块导入
# -   动态规划是一种在NLP中广泛使用的算法设计技术，通过存储以前的计算结果，避免重复计算