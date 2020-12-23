#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __init__ import *

#%%######### PARAMETERS          ############################################
p_train_val_test = [70,20,10]

# PREPROCESS DATA
p_clean_outliers = 2 # 0 = no cleaning
p_boxcox = False
p_natural_log = True
p_difference = 7
p_seasonal_decompose = False

#############################################################################
#############################################################################
###################### 1. DATA ANALYSIS                ######################
#############################################################################
#############################################################################

#%%######### LOAD DATA           ############################################

data = load_training_data('train.csv')
list_of_series = data.columns

#%%######### ANALYSE             ############################################
# Plot all series in one plot
plot_all_series(data, list_of_series)

# Clustering of time series in 4 clusters
series_classes = cluster_time_series(data.filter(regex='series'),number_of_clusters=4,seed=7,method="euclidean")

# Analyse data of the first series in each class
for i in range(max(series_classes)+1):
    selected_series, series_data = create_series_data(data,np.where(series_classes==i)[0][0]+1)
    full_monthy_plot(series_data,activated_plots=[1,0,0,1,0])
    
"""
Seasonality is the same in all classes. However the absolute values differ 
greatly between the time series 
"""    

#%%######### LOAD SERIE          ############################################

selected_series, series_data = create_series_data(data,1)

#%%######### VISUALISE           ############################################
full_monthy_plot(series_data,activated_plots=[1,1,1,1,1])

"""
We can clearly see a monthly seasonality for the day of week and for the 
months. However for monthly seasonality we do not have enough data
"""

#############################################################################
#############################################################################
###################### 2. PREPROCESSING                ######################
#############################################################################
#############################################################################



#%%######### CHECK DATES         ############################################
"""
TO BE WRITTEN....
"""


#%%######### REMOVE OUTLIERS     ############################################
series_data['data'] = remove_outliers(series_data['data'],
                                      max_iterations = p_clean_outliers,
                                      verbose=True)

#%%######### BOXCOX              ############################################
if p_boxcox:
    x, opt_lambda = boxcox(series_data['data'])
    series_data['boxcox'] = x    
    full_monthy_plot(series_data,
                     column_name="boxcox",
                     activated_plots=[1,1,1,1,1])


    # Check if this works for every time series
    data2 = data
    for series_nbr in range(data2.filter(regex='series').shape[1]):
        x, opt_lambda = boxcox(data2['series-{}'.format(series_nbr+1)]+1)
        data2['series-{}'.format(series_nbr+1)] = x
    plot_all_series(data2.filter(regex='series'), [serie for serie in list_of_series if serie[:6]=="series"])
    plot_all_series(data.filter(regex='series'), [serie for serie in list_of_series if serie[:6]=="series"])
    
    for i in range(72):
        if sum(data2['series-{}'.format(i+1)]>1000)>0:
            print(i+1)

"""
BOXCOX transform doesn't work for all series, serie 35,36,67 and 68 pose a problem
"""

#%%######### NATURAL LOG         ############################################
if p_natural_log:
    series_data['log1p'] = np.log(series_data['data']+1) 
    full_monthy_plot(series_data,
                     column_name="log1p",
                     activated_plots=[1,1,1,1,1])

#%%######### APPLY DIFFERENCING  ############################################
if p_difference:
    series_data['seasonal_diff'] = series_data["log1p"].diff(p_difference)
    full_monthy_plot(series_data,
                     column_name="seasonal_diff",
                     activated_plots=[1,1,1,1,1])

#%%######### DECOMPOSE / STL     ############################################

if p_seasonal_decompose:
    from statsmodels.tsa.seasonal import seasonal_decompose
    seasonal_decompose(series_data["boxcox"],period=11).plot()

#############################################################################
#############################################################################
###################### 3. MODEL FITTING                ######################
#############################################################################
#############################################################################

#%%######### EMPTY               ############################################
#decomposition
  
stl = STL(x, period = 7, robust = True)
  result = stl.fit()

  #predict seasonality
  #average of seasonality on a seasonal lag
  n=len(result.seasonal)
  #s_lag = seasonal lag
  s=7 #seasonality
  mat=np.zeros((s, s_lag))
  for j in range(0,s_lag):
    mat[:,j]=[result.seasonal[n-i-j*s] for i in range(1,s+1)]
  X=np.flip(np.mean(mat, axis=1))
  seasonal_prediction=np.concatenate((X,X,X),axis=None)

  #predict trend
  #compute moving average of gradient and extrapolate with it
  #t_lag = trend lag
  g=np.gradient(result.trend[-t_lag:])
  #alpha = movering average coefficient
  g_ma=0
  for i in range(0,len(g)):
    g_ma=(1-alpha)*g_ma+alpha*g[i]
  trend_prediction = []
  trend_prediction.append(result.trend[-1]+g_ma) #option1 : from result.trend
  #trend_prediction.append(DT[-1]+g_ma) #option2 : from data
  for i in range(1,horizon): 
    trend_prediction.append(trend_prediction[-1]+g_ma)
  
  print(name)

  #total predict + unboxcox
  Prediction = trend_prediction+seasonal_prediction
  #Prediction = (np.power((Prediction * opt_lambda) + 1, 1 / opt_lambda))
  Prediction = pd.DataFrame(Prediction)

  #add date
  start_test_dt = DT.index[-1] + dt.timedelta(days=1)
  end_test_dt = start_test_dt + dt.timedelta(days = horizon - 1)
  Prediction.index = pd.date_range(start_test_dt, end_test_dt)
  Prediction.columns=[name]
  Prediction[name]=Prediction[name].astype('int64')
  

#############################################################################
#############################################################################
###################### 4. POST PROCESSING              ######################
#############################################################################
#############################################################################

#%%######### EMPTY               ############################################


#############################################################################
#############################################################################
###################### 5. EVALUATE MODEL               ######################
#############################################################################
#############################################################################

#%%######### EMPTY               ############################################


#############################################################################
#############################################################################
###################### 6. GRID SEARCH BEST COMBO       ######################
#############################################################################
#############################################################################

#%%######### EMPTY               ############################################






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

