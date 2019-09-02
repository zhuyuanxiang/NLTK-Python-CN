import nltk

class IndexedText(object):
    def __init__(self, stemmer, text):
        """
        Args:
            stemmer:
            text:
        """
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width = 40):
        """
        Args:
            word:
            width:
        """
        key = self._stem(word)
        wc = int(width / 4)
        for i in self._index[key]:
            print(wc,i-wc,i)
            print(self._text[i - wc:i])
            lcontext = ' '.join(self._text[i - wc:i])
            rcontext = ' '.join(self._text[i:i + wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width = width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width = width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        """
        Args:
            word:
        """
        return self._stemmer.stem(word).lower()


porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')