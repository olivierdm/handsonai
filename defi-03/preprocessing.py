#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 22:54:57 2020

@author: olivierdemeyst
"""

from __init__ import *



def load_training_data(filename):
    data = pd.read_csv(filename, index_col = "Day", parse_dates = True)
    data = data.asfreq("D")
    
    data["d"] = data.index.day.to_numpy()
    data["m"] = data.index.month.to_numpy()
    data["y"] = data.index.year.to_numpy()
    data["w"] = data.index.weekday.to_numpy()
    data["wy"] = data.index.isocalendar().week.to_numpy()
    return data

def create_series_data(data,series_number):
        assert type(series_number)in [int,np.int64]
        selected_series = 'series-{}'.format(series_number)
        series_data = data[[selected_series, "d", "m", "y", "w", "wy"]]
        series_data = series_data.rename(columns={selected_series:'data'})
        return selected_series, series_data
    
def create_series_data2(data,series_name):
        return data[series_name].rename(columns={series_name:'data'})

def cluster_time_series(data,number_of_clusters,seed,method='euclidean'):
    #https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3 
    assert method in ['euclidean','dtw']
    assert number_of_clusters>0
    if method=='euclidean':
        model = KMeans(n_clusters=number_of_clusters, verbose=True,random_state=seed)
    elif method=='dtw':
        model = KMeans(n_clusters=number_of_clusters,
                                 metric="dtw",
                                 verbose=True,
                                 max_iter_barycenter=10,
                                 random_state=seed)
    lol = model.fit_predict(data.T)
    
    plt.figure()
    for yi in range(model.n_clusters):
        plt.subplot(model.n_clusters, 1, 1 + yi)
        for serie_name in data.T[lol == yi].index:
            plt.plot(data.index,data[serie_name], alpha=.2)
        plt.plot(data.index,model.cluster_centers_[yi].ravel(), "r-")
        plt.title('Cluster {} - #{}'.format(yi,sum(model.labels_==yi)))
    plt.show()
    
    # Class sizes
    plt.figure()
    plt.bar([i for i in range(model.n_clusters)],[sum(model.labels_==i) for i in range(model.n_clusters)])
    plt.xticks([i+1 for i in range(model.n_clusters)])
    plt.title('Clustering of time series')
    plt.xlabel('Class')
    plt.ylabel('Amount of series')
    plt.show()
    
    return model.labels_



def dftest(timeseries,plot_rolling_std=True):
    dftest = ts.adfuller(timeseries,)
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure()
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    if plot_rolling_std:
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.title('Rolling Mean and Standard Deviation')
    else:
        plt.title('Rolling Mean')
    plt.legend(loc='best')
    plt.grid()
    plt.show(block=False)



def remove_outliers(series,max_iterations,verbose=False):
    for i in range(max_iterations):
        series_cleaned, outliers = clean(series)
        if verbose:
            print("Cleaning data - iteration {} - removed outliers {}".format(i+1,sum(outliers)))
        plt.subplot(max_iterations, 1, i+1)
        plt.plot(series)
        plt.plot(series.loc[outliers], 'ro')
        series = series_cleaned
        if sum(outliers)==0:
            return series
    return series


def cross_validate(series,horizon,start,step_size,order = (1,0,0),seasonal_order = (0,0,0,0),trend=None):
    '''
    Function to determine in and out of sample testing of arima model    
    
    arguments
    ---------
    series (seris): time series input
    horizon (int): how far in advance forecast is needed
    start (int): starting location in series
    step_size (int): how often to recalculate forecast
    order (tuple): (p,d,q) order of the model
    seasonal_order (tuple): (P,D,Q,s) seasonal order of model
    
    Returns
    -------
    DataFrame: gives fcst and actuals with date of prediction
    '''
    fcst = []
    actual = []
    date = []
    for i in range(start,len(series)-horizon,step_size):
        model = sm.tsa.statespace.SARIMAX(series[:i+1], #only using data through to and including start 
                                order=order, 
                                seasonal_order=seasonal_order, 
                                trend=trend).fit()
        fcst.append(model.forecast(steps = horizon)[-1]) #forecasting horizon steps into the future
        actual.append(series[i+horizon]) # comparing that to actual value at that point
        date.append(series.index[i+horizon]) # saving date of that value
    return pd.DataFrame({'fcst':fcst,'actual':actual},index=date)