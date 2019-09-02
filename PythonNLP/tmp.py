

import nltk
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import show
from nltk.corpus import state_union

#cfdist1
cfd = nltk.ConditionalFreqDist(
    (word, fileid[:4])
    for fileid in state_union.fileids()
    for w in state_union.words(fileid)
    for word in ['men', 'women', 'people']
    if w.lower().startswith(word))
cfd.plot()


for loc, spine in cfd.spines.items():
    if loc in ['left','bottom']:
        spine.set_position(('outward',0)) # outward by 0
    elif loc in ['right','top']:
        spine.set_color('none') # don't draw spine
    else:
        raise ValueError('unknown spine location: %s'%loc)
