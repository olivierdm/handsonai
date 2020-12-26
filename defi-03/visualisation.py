#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:36:11 2020

@author: olivierdemeyst
"""
from __init__ import *

def plot_all_series(data,list_of_series,show_legend=False,title='Plot of series data'):
    fig, ax = plt.subplots()
    for serie in list_of_series:
        data[serie].plot(ax=ax,label=serie)
    ax.set(xlabel='time', ylabel='views',
           title=title)
    if show_legend:
        ax.legend()
    plt.show()

def run_sequence_plot(x, y, title, xlabel="time", ylabel="series"):
    plt.plot(x, y, 'k-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3);

def violin_plot(series_data,x_values):
    fig, ax = plt.subplots(len(x_values),1)
    for i,x_value in enumerate(x_values):
        sns.violinplot( ax = ax[i], x = series_data[x_value],y = series_data["data"] )
        ax[i].set_title("Frequency: "+x_value)
        fig.show()
        
        
def acf_pacf_plot(series_data,lags):
    fig, ax = plt.subplots(3,1)
    series_data.plot(ax=ax[0])
    ax[0].set_title("Data")
    fig.show()
    sm.tsa.graphics.plot_acf(series_data.dropna(),zero=False,ax=ax[1],lags=lags)
    ax[1].set_title("ACF")
    fig.show()
    sm.tsa.graphics.plot_pacf(series_data.dropna(),zero=False,ax=ax[2],lags=lags)
    ax[2].set_title("Partial ACF")
    fig.show()
    
    
def fft_plot(series_data,sampling_rate):
    # Frequency and sampling rate
    Fs = sampling_rate # sampling rate
    # Perform Fourier transform using scipy
    y_fft = fftpack.fft(series_data.dropna())
    # Plot data
    n = len(series_data)
    fr = Fs/2.0 * np.linspace(0,1,int(n/2))
    y_m = 2/n * abs(y_fft[0:np.size(fr)])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.stem(fr, y_m) # plot freq domain
    fig.show()
    return y_m


def full_monthy_plot(series_data,column_name='data',
                     violin_param=['w','d','m'],
                     acf_lags=30,
                     fft_FS=360*2,
                     activated_plots=[1,1,1,1,1],
                     plot_rolling_std=False):
    """
    Parameters
    ----------
    series_data : TYPE
        DESCRIPTION.
    column_name : TYPE
        DESCRIPTION.
    activated_plots : 5 element boolean array
        [DF-test,Violin plot,histogram,acf/pacf,fft]
    violin_param : TYPE
        DESCRIPTION.
    acf_lags : TYPE
        DESCRIPTION.
    fft_FS : TYPE
        DESCRIPTION.
    plot_rolling_std : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    #warnings.filterwarnings("ignore")
    # Dickie_Fuller test
    if activated_plots[0]==1:
        dftest(series_data[column_name].dropna(),plot_rolling_std=plot_rolling_std)
    
    # Seasonality analysis via violin plot
    if activated_plots[1]==1:
        violin_plot(series_data,violin_param)
    
    # Histogram
    if activated_plots[2]==1:
        fig, ax = plt.subplots(1,1)
        series_data[column_name].hist(ax=ax)
        fig.show()
    
    # Plot data and (Partial) Auto correlation
    if activated_plots[3]==1:
        acf_pacf_plot(series_data[column_name],acf_lags)
    
    
    # FFT plot
    if activated_plots[4]==1:
        try:
            fft_plot(series_data[column_name],fft_FS)
        except:
            pass
    
    #warnings.filterwarnings("default")

def plot_predictions(training_data,test_data,predictions,title="Predictions",size=[5,2]):
    fig, ax = plt.subplots(size[0],size[1])
    fig.suptitle(title)
    fig2, ax2 = plt.subplots(size[0],size[1])
    fig2.suptitle(title+" (ACF of Error)")
    fig3, ax3 = plt.subplots(size[0],size[1])
    fig3.suptitle(title+" (pACF of Error)")
    #Select 10 series to plot
    selected_series = [i*len(training_data.filter(regex="^series").columns)//(size[0]*size[1]) for i in range(1,size[0]*size[1]+1)]
    for j in range(size[0]):
        for k in range (size[1]):
            training_data[:]["series-{}".format(selected_series[j*size[1]+k])].plot(color='black',ax=ax[j,k])
            test_data[:]["series-{}".format(selected_series[j*size[1]+k])].plot(color='black',alpha=.5,ax=ax[j,k])
            predictions[:]["series-{}".format(selected_series[j*size[1]+k])].plot(color='red',ax=ax[j,k])
            
            # Residual data
            residual = test_data[:]["series-{}".format(selected_series[j*size[1]+k])]-predictions[:]["series-{}".format(selected_series[j*size[1]+k])]
            # ACF
            sm.tsa.graphics.plot_acf(residual.dropna(),zero=False,ax=ax2[j,k],lags=9)
            # pACF
            sm.tsa.graphics.plot_pacf(residual.dropna(),zero=False,ax=ax3[j,k],lags=9)
            
            
            
            
            
