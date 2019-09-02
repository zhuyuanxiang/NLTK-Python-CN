
import nltk

def unusual_words(text):
    """
    Args:
        text:
    """
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

def content_fraction(text):
    """
    Args:
        text:
    """
    stopwords=nltk.corpus.stopwords.words('english')
    content=[w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)