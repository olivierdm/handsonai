# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#pip3 uninstall scikit-learn
#pip3 install scikit-learn==0.22
#pip3 install matplotlib pmdarima seaborn tslearn sklearn supersmoother


import sys, os
import pmdarima as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from scipy import fftpack
from scipy.stats import boxcox
from scipy.special import inv_boxcox


import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf 
from statsmodels.graphics.tsaplots import plot_acf

#from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans


#os.chdir('data')
#from colorsetup import colors, palette
#sns.set_palette(palette)


from main.utils.utils_methods import *
from main.utils.utils import *
from preprocessing import *
from help_functions import *
from visualisation import *

plt.rcParams['figure.figsize'] = [10, 5]


