# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0406.py
@Version    :   v0.1
@Time       :   2020-11-18 15:33
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""


# 4.6 程序开发(P169)
# 4.6.1 Python 模块的结构(P170)
# -   每个py文件都需要有注释头，包括：模块标题 和 作者信息。
# -   模块级的 docstring，三重引号的多行字符串
# -   模块需要的所有导入语句，然后是所有全局变量，接着是组成模块主要部分的一系列函数的定义
#
# 注：Python 3 不支持 __file__

# 4.6.2 多模块程序(P171)
# -   将工作分成几个模块
# -   使用 import 语句访问其他模块定义的函数
# -   保持各个模块的简单性，并且易于维护

# 4.6.3 误差源头(P173)
# 错误代码，result 只会被初始化一次
def find_words(text, wordlength, result=[]):
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# 重复执行得到错误的结果，因为result默认不会每次调用都初始化为空列表
string = ['omg', 'teh', 'lolcat', 'sitted', 'on', 'teh', 'mat']
print("string= ", string)
print("find_words(string, 3)= ")
print(find_words(string, 3))
print("find_words(string, 2, ['ur'])= ")
print(find_words(string, 2, ['ur']))
print("find_words(string, 3)= ")
print(find_words(string, 3))
print("find_words(string, 2)= ")
print(find_words(string, 2))


# 使用None为占位符，None 是不可变对象，就可以正确初始化
# 不可变对象包含：整型、浮点型、字符串、元组
def find_words(text, wordlength, result=None):
    if not result:
        result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# 使用与默认值不同类型的对象作为默认值也可以
def find_words(text, wordlength, result=()):
    if result == ():
        result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result


# 使用与默认值不同类型的对象作为默认值也可以
def find_words(text, wordlength, result=object):
    if result is object:
        result = []
    for word in text:
        if len(word) == wordlength:
            result.append(word)
    return result

# 4.6.4 调试技术(P173)
# 使用IDE编辑工具就不用命令行调试器了

# 4.6.5 防御性编程(P174)
# 测试驱动开发
