# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0401.py
@Version    :   v0.1
@Time       :   2020-11-16 12:45
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
# Chap4 编写结构化的程序
# 1.  怎样才能写出结构良好，可读性强的程序，从而方便重用？
# 2.  基本的结构块，例如：循环、函数和赋值是如何执行的？
# 3.  Python 编程的陷阱还有哪些，如何避免它们？

# 4.1. 回到Python语言的基础
# 4.1.1 赋值

# 字符串赋值
foo = 'Monty'
bar = foo
print('foo= ', foo, ';\t', 'bar= ', bar)
foo = 'Python'
print('foo= ', foo, ';\t', 'bar= ', bar)

# 列表赋值
foo = ['Monty', 'Python']
bar = foo  # bar 对 foo 只是引用，所以 foo 的改变也会影响到 bar 的值
print('foo= ', foo, ';\t', 'bar= ', bar)
foo[1] = 'Bodkin'
print('foo= ', foo, ';\t', 'bar= ', bar)

empty = []
nested = [empty, empty, empty]
print('empty= ', empty, ';\t', 'nested= ', nested)
nested[1].append('Python')
print('empty= ', empty, ';\t', 'nested= ', nested)

nested = [[]] * 3
nested[1].append('Python')
print('nested= ', nested)
nested[1] = ['Monty']
print('nested= ', nested)
nested[1].append('Monty Python')
print('nested= ', nested)
nested[0].append('Little Python')
print('nested= ', nested)

# 复制链表（三种拷贝）
import copy

# 1:直接复制（用等于号），只复制引用

foo = ['Monty', 'Python', [1, 2]]
bar = foo
print('foo= ', foo, ';\t', 'bar= ', bar)
foo[1] = 'Bodkin'
foo[2][0] = 3
print('foo= ', foo, ';\t', 'bar= ', bar)

# 2: shadow copy浅拷贝，只复制浅层结构和深层次的引用
foo = ['Monty', 'Python', [1, 2]]
bar = copy.copy(foo)
print('foo= ', foo, ';\t', 'bar= ', bar)
foo[1] = 'Bodkin'
foo[2][0] = 3
print('foo= ', foo, ';\t', 'bar= ', bar)

# 3: deep copy 深拷贝，不复制任何引用，只复制结构
foo = ['Monty', 'Python', [1, 2]]
bar = copy.deepcopy(foo)  # 复制 foo 的结构，而不复制引用
print('foo= ', foo, ';\t', 'bar= ', bar)
foo[1] = 'Bodkin'
foo[2][0] = 3
print('foo= ', foo, ';\t', 'bar= ', bar)

# 4.1.2 等式
size = 2
python = ['Python']
snake_nest = [python] * size
snake_nest.insert(0, ['Python'])
print(snake_nest)
print(snake_nest[0] == snake_nest[1] == snake_nest[2])
print(snake_nest[1] is snake_nest[2])
print(snake_nest[1] is snake_nest[2] is snake_nest[0])

import random

size = 3
position = random.choice(range(size))
snake_nest[position] = ['Python']
print(snake_nest)
print(snake_nest[0] == snake_nest[1] == snake_nest[2])
print(snake_nest[0] is snake_nest[1] is snake_nest[2])
snake_nest.insert(0, ['Monty Python'])
print(snake_nest)

# id(snake) 字符串签名编码
[id(snake) for snake in snake_nest]

# 4.1.3 条件语句
# 条件判别语句
mixed = ['cat', '', ['dog'], []]
for element in mixed:
    if element:  # 非空字符串或非空链表判为真；空字符串或空链表判为假
        print(element)

# 判决短路，第一个条件满足以后，后面的就不再判别就不再执行
animals = ['cat', 'dog']
if 'cat' in animals:
    print(1)
elif 'dog' in animals:
    print(2)

sent = ['No', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '.']
print("all()= ", all(len(w) > 4 for w in sent))  # all()检查全部满足条件
print("any()= ", any(len(w) > 4 for w in sent))  # any()检查部分满足条件
