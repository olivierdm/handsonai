#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:15:52 2020

@author: olivierdemeyst
"""

#Reconstruct original data
def undo_differencing(training_data,forecast_data,difference_data,shift_number):
    """

    Parameters
    ----------
    training_data : pandas series
        DESCRIPTION.
    forecast_data : pandas series
        DESCRIPTION.
    difference_data : pandas series
        DESCRIPTION.
    shift_number : integer
        DESCRIPTION.

    Returns
    -------
    reconstructed data

    """
    if type(forecast_data)!=pd.core.series.Series:
        input_data = training_data
    else:
        input_data = pd.concat([training_data,forecast_data])
        difference_data = pd.concat([difference_data,pd.Series(0,index=forecast_data.index)])

    reconstructed_data = pd.Series(0,index=input_data.index)
    reconstructed_data = input_data + difference_data.shift(shift_number)
    if type(forecast_data)==pd.core.series.Series:
        for i in range(len(training_data)+7,len(reconstructed_data)):
            reconstructed_data[i] = reconstructed_data[i]+reconstructed_data[i-7]
        #reconstructed_data = reconstructed_data + difference_data_forecast.shift(shift_number)
    
    return reconstructed_data