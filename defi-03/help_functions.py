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

def calculate_smape_df(reference_data,forecast_data):
    results = {}
    for series_name in forecast_data.filter(regex='^series').columns:
        results[series_name] = smape(reference_data[series_name],forecast_data[series_name])
    return results


def export_predictions_csv(predictions,file,HORIZON=21):
    ids_list = []
    predictions_list = []
    get_id_list = lambda series: ["s{}h{}".format(series.split("-")[1],i) for i in range(1,HORIZON+1)]
    for series_name in predictions.filter(regex='^series').columns:
        ids_list+=get_id_list(series_name)
        predictions_list+=list(predictions[series_name])
    pd.DataFrame(np.array([ids_list,predictions_list]).T,
                 columns=['Id','forecasts']).to_csv(file, 
                                                   index=False)