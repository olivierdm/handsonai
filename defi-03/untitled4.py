#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:24:35 2020

@author: olivierdemeyst
"""


1. DATA ANALYSIS
    PLOTS and TEST to analyse data.
        - Running average mean/std
        - Violin plot to see seasonality
        - Histogram of data
        - ACF/pACF
        - FFT?
        
    Comparison of all series and try to group similar series
    
2. PRE-PROCESSING
    - Je propose d'utiliser le minimum de differente preprocessing methodes. 
    
    - Pour le moment j'ai utilise le boxcox + differencing (shift 7) pour obtenir un stationary dataset
      Si pour les simples methodes (mean, naive, snaive) le seasonal_decompose marche mieux, nous pouvons garder cette methode
    
    - PLOT ANALYSIS stationary data (+ DF test)
    
3. MODEL FITTING
    - mean
    - naive
    - snaive
        7 days
        14 days?
        1 year?
    - ARMA
    - SARIMAX
    - CNN ??
    - RNN ??

4. POST PROCESSING
    - UNDO DIFFERENCING
    - INV BOXCOX
    

4. VERIFY DATA
    - SMAPE
    - PLOT ANALYSIS residiual data
    
5. GRID SEARCH BEST COMBINATION OF MODELS
    - On selection of series to win time
    
    
    
    
    
NICE CODEEEEE


#rolling window
series_data['data'].rolling(30).mean().plot()



