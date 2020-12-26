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

## EXP SMOOTHING ############################


from statsmodels.tsa.api import ExponentialSmoothing

model_1 = ExponentialSmoothing(train_1,damped=True,
                              trend="additive",
                              seasonal=None,
                              seasonal_periods=None).fit(optimized=True)

##############################

model = sm.tsa.ARMA(series_data['seasonal_diff'].dropna(), (1, 0)).fit(trend='nc', disp=0)
model.params


##############################

sar3 = sm.tsa.statespace.SARIMAX(series_data['boxcox'], 
                                order=(1,0,0), 
                                seasonal_order=(0,7,0), 
                                trend='c').fit()

sm.tsa.graphics.plot_pacf(sar3.resid[sar3.loglikelihood_burn:]);
sar3.plot_diagnostics();

##############################

    
#%%######### DECOMPOSE / STL     ############################################

if p_seasonal_decompose:
    from statsmodels.tsa.seasonal import seasonal_decompose
    seasonal_decompose(series_data["boxcox"],period=11).plot()








training_data = series_data['seasonal_diff'][:int(len(series_data)*0.7)]#data[[selected_series]][:int(train_val_test[0]*len(data)/100)]#series_data['seasonal_diff'].dropna()
test_data = series_data['seasonal_diff'][int(len(series_data)*0.7):]#data[[selected_series]][int(train_val_test[0]*len(data)/100):]

# Create auto_arima model
model1 = pm.auto_arima(training_data.dropna(),
                      seasonal=True, m=7,
                      d=0, D=1,
                      start_p=0, max_p=0,
                      start_q=0, max_q=0,
                      start_P=3, max_P=6,
                      start_Q=0, max_Q=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True)
                       
# Print model summary
print(model1.summary())

# Create model object
order = (1,0,0)#model1.order
seasonal_order = (5,1,2,14)#model1.seasonal_order
model = sm.tsa.statespace.SARIMAX(training_data, 
                order=order, 
                seasonal_order=seasonal_order, 
                trend='c')
# Fit model
results = model.fit()

# Plot common diagnostics
results.plot_diagnostics()
plt.show()
plt.close()

# Create forecast object
forecast_object = results.get_forecast(steps=len(test_data))

# Extract prediction mean
mean = forecast_object.predicted_mean

# Extract the confidence intervals
conf_int = forecast_object.conf_int()

# Extract the forecast dates
dates = mean.index

# Print last predicted mean
print(mean.iloc[-1])

# Print last confidence interval
print(conf_int.iloc[-1])

## Validating Forecast
#pred = results.get_prediction(start=pd.to_datetime('2016-12-01'), dynamic=False)
#pred_ci = pred.conf_int()
ax = pd.concat([training_data['2016-':],test_data]).plot(label='observed')
forecast_object.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(conf_int.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()


#Reconstruct data
reconstructed_data = undo_differencing(training_data,
                                       forecast_object.predicted_mean,
                                       series_data['boxcox'][:len(training_data)],
                                       7)

run_sequence_plot(reconstructed_data.index, reconstructed_data,title="Reconstructed data of "+selected_series)
if opt_lambda==0:
    reconstructed_data = np.exp(reconstructed_data)
else:
    reconstructed_data = (opt_lambda*reconstructed_data+1)**(1/opt_lambda)
run_sequence_plot(reconstructed_data.index, reconstructed_data,title="Reconstructed data of "+selected_series)


#Calculate SMAPE
smape_sarima = smape(test_data,reconstructed_data[test_data.index[0]:])

##############################




#inv_boxcox(series_data['boxcox'], opt_lambda)