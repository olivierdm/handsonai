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
import warnings, logging
from datetime import timedelta
import glob
import re

from scipy import fftpack
from scipy.stats import boxcox
from scipy.special import inv_boxcox


import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.compat import lzip

from fbprophet import Prophet

from pmdarima.arima import auto_arima

#from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans


#os.chdir('data')
#from colorsetup import colors, palette
#sns.set_palette(palette)


from main.utils.utils_methods import *
from main.utils.utils import *
from main.module.cnn_dilated import *
from main.module.mlp_multioutput import *
from main.module.rnn_encoder_decoder import *
from main.module.rnn_vector_output import *
from preprocessing import *
from help_functions import *
from visualisation import *
from methods import *

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from main.module.mlp_multioutput import mlp_multioutput
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dense
from keras.layers import GRU, RepeatVector, TimeDistributed, Flatten, Input
from keras import regularizers



plt.rcParams['figure.figsize'] = [10, 10]


