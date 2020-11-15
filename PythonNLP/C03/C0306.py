# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   NLTK-Python-CN
@File       :   C0306.py
@Version    :   v0.1
@Time       :   2020-11-15 17:39
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   
@理解：
"""
import nltk

# 3.6 规范化文本(P115)
# 词干提取器错误很多

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  
 Supreme executive power derives from a mandate from the masses, 
 not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
print(tokens[:13])

# Porter 词干提取器
porter = nltk.PorterStemmer()
stem_porter_list = [
        porter.stem(t)
        for t in tokens
]
print(stem_porter_list[:13])

# Lancaster 词干提取器
lancaster = nltk.LancasterStemmer()
stem_lancaster_list = [
        lancaster.stem(t)
        for t in tokens
]
print(stem_lancaster_list[:13])


# Ex3-1 使用词干提取器索引文本
class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index(
                (self._stem(word), i)
                for (i, word) in enumerate(text)
        )

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width / 4)
        for i in self._index[key]:
            l_context = ' '.join(self._text[i - wc:i])
            r_context = ' '.join(self._text[i:i + wc])
            l_display = '{:>{width}}'.format(l_context[-width:], width=width)
            r_display = '{:{width}}'.format(r_context[:width], width=width)
            print(l_display, r_display)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()


grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('em')

# 3.6.2 词形归并(P117)
# WordNet词形归并器将会删除词缀产生的词，即将变化的单词恢复原始形式，例如：women转变成woman

wnl = nltk.WordNetLemmatizer()
wnl_lemma_list = [
        wnl.lemmatize(t)
        for t in tokens
]
print(wnl_lemma_list)

