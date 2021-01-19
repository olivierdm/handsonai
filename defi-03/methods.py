#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 20:53:31 2020

@author: olivierdemeyst
"""
from __init__ import *


def avg_method_pred(training_data,HORIZON):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name] = training_data[series_name].mean()
    return data_predictions

def naive_method_pred(training_data,HORIZON):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name] = training_data[series_name][-1]
    return data_predictions
"""
def snaive_method_pred(training_data,HORIZON):
    data_predictions = pd.DataFrame(data.filter(regex="^(?!series)")[-HORIZON:],index=data.index[-HORIZON:],columns=data.filter(regex="^(?!series)").columns)
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name]=0
        for i,week_day in enumerate(data_predictions["w"]):
            tmp_date_last_weekday = training_data["w"].where(training_data["w"]==week_day).last_valid_index()
            data_predictions[series_name][i]=training_data[series_name][tmp_date_last_weekday]
    return data_predictions
"""

def exp_smoothing_method_pred(training_data,HORIZON,METHOD="simple",smoothing_level=.3,optimized=True,smoothing_slope=.05):
    exp_smoothing_type = METHOD #"simple"
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    data_predictions.astype(np.float)
    for series_name in training_data.filter(regex='^series').columns:
        if exp_smoothing_type == "holt":
            model = Holt(training_data[series_name])
        elif exp_smoothing_type == "simple":
            model = SimpleExpSmoothing(training_data[series_name])
        
        model._index = training_data.index
        
        if exp_smoothing_type == "holt":
            if optimized:
                fit = model.fit(optimized=True)
            else:
                fit = model.fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope)
        elif exp_smoothing_type == "simple":
            fit = model.fit(smoothing_level=smoothing_level)
    
        
        #pred = fit.forecast(HORIZON)
        data_predictions[series_name] = fit.forecast(HORIZON)
        
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(training_data.index[-5*HORIZON:], training_data['series-1'].values[-5*HORIZON:])
        ax.plot(prediction_reference_data.index, prediction_reference_data['series-1'].values, color="gray")
        for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('#ff7823','#3c763d','c')):
            ax.plot(training_data.index[-5*HORIZON:], f.fittedvalues[-5*HORIZON:], color=c)
            if exp_smoothing_type == "simple":
                ax.plot(prediction_reference_data.index, p, label="alpha="+str(f.params['smoothing_level'])[:3], color=c)
            elif exp_smoothing_type == "holt":
                ax.plot(prediction_reference_data.index, p, label="alpha="+str(f.params['smoothing_level'])[:4]+", beta="+str(f.params['smoothing_trend'])[:4], color=c)
        if exp_smoothing_type == "holt":
            plt.title("Holt's Exponential Smoothing")
        elif exp_smoothing_type == "simple":
            plt.title("Simple Exponential Smoothing")
        plt.legend();
        """
    return data_predictions

def snaive_method_pred(training_data,HORIZON):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name]=0
        for i,day in enumerate(data_predictions.index):
            multiplier = (day-training_data.index[-1])//timedelta(days=7) + 1
            data_predictions[series_name][i]=training_data[series_name][day-timedelta(days=7)*multiplier]
    return data_predictions

def snaive_year_method_pred(training_data,HORIZON):
    moving_average_window = 28
    rescale_look_back = 3*30
    
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        #new = training_data[series_name]['2017-03-02':'2017-08-20'].values
        old_avg = training_data[series_name].rolling(window=moving_average_window).mean()
        scale_factor = training_data[series_name][training_data.index[-rescale_look_back:]].values.mean()/training_data[series_name][training_data.index[-rescale_look_back:]-timedelta(days=365)].values.mean()
        data_predictions[series_name] = old_avg[data_predictions.index-timedelta(days=365)].values*scale_factor
    return data_predictions


def snaive_avg_method_pred(training_data,HORIZON,history_mean):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    data_predictions.astype(np.float)
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name]=0.0
        for i,day in enumerate(data_predictions.index):
            multiplier = (day-training_data.index[-1])//timedelta(days=7) + 1
            calculated_mean = training_data[series_name][day-timedelta(days=7)*(multiplier)]
            for k in range(1,history_mean//7):
                calculated_mean += training_data[series_name][day-timedelta(days=7)*(multiplier+k)]
            calculated_mean = calculated_mean/(history_mean//7)
            data_predictions[series_name][i]=float(calculated_mean)
    return data_predictions

def snaive_median_method_pred(training_data,HORIZON,history_mean):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    data_predictions.astype(np.float)
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name]=0.0
        for i,day in enumerate(data_predictions.index):
            multiplier = (day-training_data.index[-1])//timedelta(days=7) + 1
            calculated_mean = np.array(training_data[series_name][day-timedelta(days=7)*(multiplier)])
            for k in range(1,history_mean//7):
                calculated_mean = np.append(calculated_mean,(training_data[series_name][day-timedelta(days=7)*(multiplier+k)]))
            calculated_mean = np.median(calculated_mean)
            data_predictions[series_name][i]=float(calculated_mean)
    return data_predictions

def snaive_exp_smoothing_method_pred(training_data,HORIZON,METHOD="simple",smoothing_level=.3,optimized=True,smoothing_slope=.05):
    """
    Method hardcoded for weekday seasonality

    """
    
    
    exp_smoothing_type = METHOD #"simple"
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    data_predictions.astype(np.float)
    for i in range(7):
        for series_name in training_data.filter(regex='^series').columns:
            try:
                data_predictions[series_name].shape
            except:
                data_predictions[series_name]=0.0
            if exp_smoothing_type == "holt":
                model = Holt(training_data[series_name][training_data.index.dayofweek==i])
            elif exp_smoothing_type == "simple":
                model = SimpleExpSmoothing(training_data[series_name][training_data.index.dayofweek==i])
            
            model._index = training_data[training_data.index.dayofweek==i].index
            
            if exp_smoothing_type == "holt":
                if optimized:
                    fit = model.fit(optimized=True)
                else:
                    fit = model.fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope)
            elif exp_smoothing_type == "simple":
                fit = model.fit(smoothing_level=smoothing_level)
        
            
            #pred = fit.forecast(HORIZON)
            data_predictions[series_name][data_predictions.index.dayofweek==i] = fit.forecast(HORIZON//7)
    
    return data_predictions

"""
def snaive_avg_method_pred(training_data,HORIZON,history_mean):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        data_predictions[series_name]=0
        calculated_mean = 0
        for i in range(1,history_mean//7):
            calculated_mean+=training_data[series_name][day-timedelta(days=7)*multiplier]
        data_predictions[series_name][i]=training_data[-history_mean:][series_name].where(training_data["w"]==week_day).mean()
    return data_predictions
"""

def stl_arima_method_pred(training_data,HORIZON,arima_order):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        #elec_equip.index.freq = elec_equip.index.inferred_freq
        stlf = STLForecast(training_data[series_name], ARIMA, model_kwargs=dict(order=arima_order, trend="c"))
        stlf_res = stlf.fit()
        data_predictions[series_name] = stlf_res.forecast(HORIZON)
    return data_predictions

def snaive_decomp_method_pred(training_data,HORIZON,opt_lambdas):  
    s_lag = HORIZON
    t_lag = 7
    alpha = 0.5
    
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        
        stl = STL(training_data["boxcox-"+series_name],
                  period = 7,
                  robust = True, 
                  trend_deg=0)
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
        for i in range(1,HORIZON): 
          trend_prediction.append(trend_prediction[-1]+g_ma)
        
        
        #total prediction
        data_predictions[series_name]=inv_boxcox(trend_prediction+seasonal_prediction,opt_lambdas[series_name])
    return data_predictions


def snaive_decomp_method_mod_pred(training_data,HORIZON,opt_lambdas):  
    s_lag = HORIZON
    t_lag = 7
    alpha = 0.5
    
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^series').columns:
        
        stl = STL(training_data["boxcox-"+series_name],
                  period = 7,
                  robust = True, 
                  trend_deg=0)
        result = stl.fit()
        
        #predict seasonality
        """
        #average of seasonality on a seasonal lag
        n=len(result.seasonal)
        #s_lag = seasonal lag
        s=7 #seasonality
        mat=np.zeros((s, s_lag))
        for j in range(0,s_lag):
          mat[:,j]=[result.seasonal[n-i-j*s] for i in range(1,s+1)]
        X=np.flip(np.mean(mat, axis=1))
        seasonal_prediction=np.concatenate((X,X,X),axis=None)
        """
        seasonal_prediction=np.array(snaive_avg_method_pred(pd.DataFrame(result.seasonal,index=training_data.index).rename(columns={'season':series_name}),
                               HORIZON, 
                               35)[series_name])
        
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
        for i in range(1,HORIZON): 
          trend_prediction.append(trend_prediction[-1]+g_ma)
        
        
        #total prediction
        data_predictions[series_name]=inv_boxcox(trend_prediction+seasonal_prediction,opt_lambdas[series_name])
    return data_predictions

def fb_prophet_method_pred(training_data,HORIZON,opt_lambdas=None, additive_model=False):
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    data_predictions['ds']=data_predictions.index
    training_data['ds']=training_data.index
    for series_name in training_data.filter(regex='^series').columns:
        print('Analysing '+series_name)
        if additive_model:
            training_data['y']=training_data['boxcox-'+series_name]
            model = Prophet(seasonality_mode="additive",
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=True).add_seasonality(
                                name="daily",
                                period = 1,
                                fourier_order = 5,
                                prior_scale=1).add_seasonality(
                                name="weekly",
                                period = 7,
                                fourier_order = 20,
                                prior_scale=10).add_seasonality(
                                name="yearly",
                                period = 365.25,
                                fourier_order = 10,
                                prior_scale=12).add_seasonality(
                                name="quarterly",
                                period = 365.25/4,
                                fourier_order = 5,
                                prior_scale=1)
        else:
            training_data['y']=training_data[series_name]
            # define the model
            model = Prophet()
        
        # fit the model
        model.fit(training_data[['ds','y']])
        if additive_model:
            data_predictions[series_name] = inv_boxcox(np.array(model.predict(data_predictions[['ds']])['yhat']),opt_lambdas[series_name])
        else:
            data_predictions[series_name] = np.array(model.predict(data_predictions[['ds']])['yhat'])
    return data_predictions#data_predictions


def mlp_multioutput_method_pred(training_data,HORIZON,opt_lambdas,p_difference):
    # The number of lagged values.
    LAG = 30
    LATENT_DIM = 50 #5   # number of units in the RNN layer
    BATCH_SIZE = 32  # number of samples per mini-batch
    EPOCHS = 100      # maximum number of times the training algorithm will cycle through all samples
    loss = 'mse' # Loss function to be used to optimize the model parameters
    adam_learning_rate=0.01
    early_stopping_patience=100
    early_stopping_delta=0
    
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^diff-series').columns:
        series_data = pd.DataFrame(training_data[series_name][p_difference:].values,index=training_data.index[p_difference:],columns=['series'])
        
        # Data split
        n = len(series_data)
        n_train = int(0.8 * n)#(n - HORIZON - LAG))
        n_valid = n - n_train #- HORIZON - LAG #Must be minamally HORIZON+LAG = 51 to test last 21 days prediction
        n_learn = n_train + n_valid
        
        train = series_data[:n_train]
        valid = series_data[n_train:n_learn]
        #test = series_data[n_learn:n]
        
        # From time series to input-output data (also called time series embedding)
        train_inputs, valid_inputs, X_train, y_train, X_valid, y_valid, \
            = embed_data(train, valid, None, HORIZON, LAG, freq = "D", variable = 'series')
                
        #########################
        file_header = "model_" + series_name + "_mlp_multioutput"
        verbose = 0
        
        optimizer_adam = keras.optimizers.Adam(learning_rate=adam_learning_rate) 
        earlystop = EarlyStopping(monitor='val_loss', 
                                  min_delta=early_stopping_delta, 
                                  patience= early_stopping_patience)
        
        
        
        best_val = ModelCheckpoint('../work/' + file_header + '_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
        #########################
        
        model_mlp_multioutput, history_mlp_multioutput = mlp_multioutput(X_train, y_train, X_valid, y_valid, 
                                LATENT_DIM = LATENT_DIM, 
                                BATCH_SIZE = BATCH_SIZE, 
                                EPOCHS = EPOCHS, 
                                LAG = LAG, 
                                HORIZON = HORIZON, 
                                loss = loss, 
                                optimizer = optimizer_adam,
                                earlystop = earlystop, 
                                best_val = best_val,
                                verbose=verbose)
        plot_learning_curves(history_mlp_multioutput)
        
        best_epoch = np.argmin(np.array(history_mlp_multioutput.history['val_loss']))+1
        filepath = '../work/' + file_header + '_{:02d}.h5'
        model_mlp_multioutput.load_weights(filepath.format(best_epoch))

        #Generate input for prediction
        tensor_structure = {'encoder_input':(range(-LAG+1, 1), ["series"]), 'decoder_input':(range(0, HORIZON), ["series"])}
        data_inputs = TimeSeriesTensor(series_data[-LAG-HORIZON:], "series", HORIZON, tensor_structure, freq = "D")
        DT = data_inputs.dataframe
        X_test = DT.iloc[:, DT.columns.get_level_values(0)=='encoder_input']
        X_test.values[0] =training_data[series_name][-LAG:].values
        
        predictions_mlp_multioutput = model_mlp_multioutput.predict(X_test)
        data_predictions[series_name[5:]] = pd.Series(predictions_mlp_multioutput[0], index=data_predictions.index)
        data_predictions[series_name[5:]] = undo_differencing(training_data["boxcox-"+series_name[5:]][-p_difference:],data_predictions[series_name[5:]],p_difference)
        data_predictions[series_name[5:]] = inv_boxcox(data_predictions[series_name[5:]],opt_lambdas[series_name[5:]])

    return data_predictions

def mlp_recursive_method_pred(training_data,HORIZON,opt_lambdas,p_difference):
    LAG=30
    LATENT_DIM = 5   # number of units in the RNN layer
    BATCH_SIZE = 32  # number of samples per mini-batch
    EPOCHS = 100      # maximum number of times the training algorithm will cycle through all samples
    loss = 'mse'
    adam_learning_rate=0.01
    early_stopping_patience=100
    early_stopping_delta=0
    
    
    verbose = 0
    
    optimizer_adam = keras.optimizers.Adam(learning_rate=adam_learning_rate) 
    earlystop = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience= early_stopping_patience)
    
    data_predictions = pd.DataFrame(index=training_data.index[-HORIZON:]+timedelta(days=HORIZON))
    for series_name in training_data.filter(regex='^diff-series').columns:
        print("Predicting next values for "+series_name[5:])
        file_header = "model_" + series_name + "_mlp_recursive"
        best_val = ModelCheckpoint('../work/' + file_header + '_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
        #########################
        series_data = pd.DataFrame(training_data[series_name][p_difference:].values,index=training_data.index[p_difference:],columns=['series'])
        # Data split
        n = len(series_data)
        n_train = int(0.8 * n)#(n - HORIZON - LAG))
        n_valid = n - n_train #- HORIZON - LAG #Must be minamally HORIZON+LAG = 51 to test last 21 days prediction
        n_learn = n_train + n_valid
        
        train = series_data[:n_train]
        valid = series_data[n_train:n_learn]
        #test = series_data[n_learn:n]
        _, _, X_train_onestep, y_train_onestep, X_valid_onestep, y_valid_onestep = embed_data(train, valid, None, 1, LAG, freq = None, variable = 'series')
        
        model_mlp_recursive, history_mlp_recursive = mlp_multioutput(X_train_onestep, y_train_onestep, X_valid_onestep, y_valid_onestep, 
                                LATENT_DIM = LATENT_DIM, 
                                BATCH_SIZE = BATCH_SIZE, 
                                EPOCHS = EPOCHS, 
                                LAG = LAG, 
                                HORIZON = 1, 
                                loss = loss, 
                                optimizer = optimizer_adam,
                                earlystop = earlystop, 
                                best_val = best_val,
                                verbose=verbose)
        plot_learning_curves(history_mlp_recursive)
        
        best_epoch = np.argmin(np.array(history_mlp_recursive.history['val_loss']))+1
        filepath = '../work/' + file_header + '_{:02d}.h5'
        model_mlp_recursive.load_weights(filepath.format(best_epoch))
        
        #Generate input for prediction
        tensor_structure = {'encoder_input':(range(-LAG+1, 1), ["series"]), 'decoder_input':(range(0, HORIZON), ["series"])}
        data_inputs = TimeSeriesTensor(series_data[-LAG-HORIZON:], "series", HORIZON, tensor_structure, freq = "D")
        DT = data_inputs.dataframe
        X_test = DT.iloc[:, DT.columns.get_level_values(0)=='encoder_input']
        X_test.values[0] =training_data[series_name][-LAG:].values
        
        for h in range(HORIZON):
            pred = model_mlp_recursive.predict(X_test)
            X_test = pd.DataFrame(np.hstack( (np.delete(X_test.to_numpy(), 0, 1), pred) ), index = X_test.index, columns =X_test.columns)
            if h > 0:
                predictions_mlp_recursive = np.hstack( (predictions_mlp_recursive, pred) )
            else:
                predictions_mlp_recursive = pred
        #predictions_mlp_recursive = pd.DataFrame(predictions_mlp_recursive, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    
        #predictions_mlp_multioutput = model_mlp_multioutput.predict(X_test)
        data_predictions[series_name[5:]] = pd.Series(predictions_mlp_recursive[0], index=data_predictions.index)
        data_predictions[series_name[5:]] = undo_differencing(training_data["boxcox-"+series_name[5:]][-p_difference:],data_predictions[series_name[5:]],p_difference)
        data_predictions[series_name[5:]] = inv_boxcox(data_predictions[series_name[5:]],opt_lambdas[series_name[5:]])

    return data_predictions