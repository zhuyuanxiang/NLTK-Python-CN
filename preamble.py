import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from IPython.display import display
from IPython.display import set_matplotlib_formats
from cycler import cycler

set_matplotlib_formats('pdf', 'png')

# 不能正常显示，拷贝字体后需要删除"用户/.matplotlib/fontList.json"
# plt.rcParams.keys()
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['savefig.dpi'] = 300
# ----------------------------------------------------------------------
# mpl.rcParams['font.family'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# mpl.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']
# mpl.rcParams['font.size'] = 9

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)

pd.set_option('precision', 3)
# -------
# 显示所有列，默认为5
pd.set_option('display.max_columns', None)
# pd.set_option("display.max_columns", 8)
# -------
# deprecated，使用display_max_rows代替
# pd.set_option('display.height',1000)
# 显示所有行，默认为5
pd.set_option('display.max_rows', None)
# -------
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 500)
# -------
# 设置显示的宽度，默认为80
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore')

__all__ = ['np', 'nltk', 'display', 'plt', 'pd']
