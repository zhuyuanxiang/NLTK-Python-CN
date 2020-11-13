# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0203.py
@Version    :   v0.1
@Time       :   2020-11-13 9:54
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""

# Chap2 获取文本语料库 和 词汇资源
# 目的：
# 1.  什么是有用的文本语料库和词汇资源？如何使用 Python 获取它们？
# 2.  哪些 Python 结构最适合这项工作？
# 3.  编写 Python 代码时如何避免重复的工作？


# 2.3 Python 中的 代码重用
# 2.3.1 使用文本编辑器创建Python程序
# 建议使用IDE：VSCode 或者 PyCharm
# 2.3.2 如何定义函数
# 导入精确除法，现在默认是精确除法，使用截断除法需要使用 3//4
# from __future__ import division

# 2.3.3 Python 模块
# 修改 func 中的代码时，需要重启 Python Console 环境
from func import plural

print("plural('wish')= ", plural('wish'))
print("plural('fan')= ", plural('fan'))
