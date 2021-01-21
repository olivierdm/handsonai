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


assert p_boxcox!=p_natural_log

#############################################################################
#############################################################################
###################### 1. DATA ANALYSIS                ######################
#############################################################################
#############################################################################

#%%######### LOAD DATA           ############################################

orig_data = load_training_data('train.csv')
data = orig_data.copy()
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


#%%######### VISUALISE EDA        ############################################
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
if int((data.index[-1]-data.index[0])/np.timedelta64(1, 'D')+1)==len(data.index):
    print("No missing dates")
else:
    print("Missing dates detected")


#%%######### CLEAN + LOG1P/BOXCOX + DIFF(7) #################################
warnings.filterwarnings("ignore")
for series_name in data.filter(regex='series').columns:
        data[series_name] = remove_outliers(data[series_name],
                                            max_iterations = p_clean_outliers,
                                            verbose=False)
data2 = data.copy()
last_prefix = ""
if p_boxcox:
    opt_lambdas = {}
for series_name in data2.filter(regex='series').columns:
    if p_natural_log:
        data2['log-'+series_name]  = np.log(data2[series_name])
        opt_lambdas = []
        last_prefix = "log-"
    elif p_boxcox: 
        x, opt_lambda = boxcox(data2[series_name])
        data2['boxcox-'+series_name]=x
        opt_lambdas[series_name]=opt_lambda
        last_prefix = "boxcox-"
    if p_difference:
        data2['diff-'+series_name] = data2[last_prefix+series_name].diff(p_difference)

plot_all_series(data2.filter(regex='^series'), [serie for serie in list_of_series if serie[:6]=="series"],show_legend=False, title="Unprocessed data")
if p_natural_log:
    plot_all_series(data2.filter(regex='log-series'), ["log-"+serie for serie in list_of_series if serie[:6]=="series"],show_legend=False, title="Natural log of data")
if p_boxcox:
    plot_all_series(data2.filter(regex='boxcox-series'), ["boxcox-"+serie for serie in list_of_series if serie[:6]=="series"],show_legend=False, title="Boxcox of data")
if p_difference:
    plot_all_series(data2.filter(regex='diff-series'), ["diff-"+serie for serie in list_of_series if serie[:6]=="series"],show_legend=False,title="Differenced data (lag={})".format(p_difference))



#############################################################################
#############################################################################
###################### 3. MODEL FITTING                ######################
#############################################################################
#############################################################################


#%%######### NAIVE METHODS       ############################################
HORIZON = 21

prediction_reference_data = data[-HORIZON:].copy()
#training_data[training_data.filter(regex='^series').columns] = data.filter(regex='^series').rolling(7).sum()[:-HORIZON]
training_data = data[:-HORIZON].copy()

#%%


#%%
show_plot = False
#Predict only subset of series that begin with str_filter
str_filter = "series-7"

smape_arrays = {}

avg_pred = avg_method_pred(training_data, HORIZON)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,avg_pred,title="Average prediction",size=[2,2])
smape_arrays['smape_avg'] = np.array(list(calculate_smape_df(prediction_reference_data,avg_pred).values()))

naive_pred = naive_method_pred(training_data, HORIZON)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,naive_pred,title="Naive prediction",size=[2,2])
smape_arrays['smape_naive'] = np.array(list(calculate_smape_df(prediction_reference_data,naive_pred).values()))

exp_smoothing_pred = exp_smoothing_method_pred(training_data, HORIZON,METHOD="holt",optimized=True)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,exp_smoothing_pred,title="Exp smoothing prediction",size=[8,2])
smape_arrays['smape_exp_smoothing'] = np.array(list(calculate_smape_df(prediction_reference_data,exp_smoothing_pred).values()))

snaive_year_pred = snaive_year_method_pred(training_data, HORIZON)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_year_pred,title="Snaive year prediction",size=[8,2])
smape_arrays['smape_snaive_year'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_year_pred).values()))

snaive_exp_smoothing_pred = snaive_exp_smoothing_method_pred(training_data, HORIZON,METHOD="holt",optimized=False,smoothing_slope=0.018,smoothing_level=.34)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_exp_smoothing_pred,title="Snaive Exp smoothing prediction",size=[8,2])
smape_arrays['smape_snaive_exp_smoothing'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_exp_smoothing_pred).values()))

snaive_pred = snaive_method_pred(training_data, HORIZON)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_pred,title="Snaive prediction",size=[2,2])
smape_arrays['smape_snaive'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_pred).values()))

snaive_avg_pred = snaive_avg_method_pred(training_data, HORIZON, 35) #Best result for 35 days
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_avg_pred,title="Snaive average prediction",size=[8,2])
smape_arrays['smape_snaive_avg'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_avg_pred).values()))

snaive_median_pred = snaive_median_method_pred(training_data, HORIZON, 35) #Best result for 35 days
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_median_pred,title="Snaive median prediction",size=[8,2])
smape_arrays['smape_snaive_median'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_median_pred).values()))

snaive_decomp_pred = snaive_decomp_method_pred(data2[:-HORIZON],HORIZON,opt_lambdas) 
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,snaive_decomp_pred,title="Snaive decomposition prediction",size=[8,2])
smape_arrays['smape_snaive_decomp'] = np.array(list(calculate_smape_df(prediction_reference_data,snaive_decomp_pred).values()))


#%%######### FACEBOOK PROPHET     ############################################
fb_prophet_pred = fb_prophet_method_pred(training_data,HORIZON) 
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,fb_prophet_pred,title="FB Prophet prediction",size=[8,2])
smape_arrays['smape_fb_prophet'] = np.array(list(calculate_smape_df(prediction_reference_data,fb_prophet_pred).values()))

fb_prophet_mod_pred = fb_prophet_method_pred(data2[:-HORIZON],HORIZON,opt_lambdas=opt_lambdas,p_difference=p_difference,additive_model=True) 
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,fb_prophet_mod_pred,title="FB Prophet Mod prediction",size=[8,2])
smape_arrays['smape_fb_prophet_mod'] = np.array(list(calculate_smape_df(prediction_reference_data,fb_prophet_mod_pred).values()))


#%%######### SARIMA/ARIMA METHODS ###########################################

stl_arima_pred = stl_arima_method_pred(training_data,HORIZON,(2,1,1))
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,stl_arima_pred,title="STL Arima prediction",size=[2,2])
smape_arrays['smape_stl_arima'] = np.array(list(calculate_smape_df(prediction_reference_data,stl_arima_pred).values()))

    
#%%######### SARIMAX             ############################################

sarimax_pred = sarimax_method_pred(data2[:-HORIZON], HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:],prediction_reference_data,sarimax_pred,title="SARIMAX prediction",size=[4,2])
smape_arrays['smape_sarimax'] = np.array(list(calculate_smape_df(prediction_reference_data,sarimax_pred).values()))

#%%######### NETWORKS #######################################################

############ MLP MULTIOUTPUT     ############################################


mlp_multioutput_pred = mlp_multioutput_method_pred(data2[:-HORIZON].filter(regex=str_filter),HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:].filter(regex=str_filter),prediction_reference_data.filter(regex=str_filter),mlp_multioutput_pred,title="MLP multioutput prediction",size=[2,2])
smape_arrays['smape_mlp_multioutput'] = np.array(list(calculate_smape_df(prediction_reference_data.filter(regex=str_filter),mlp_multioutput_pred).values()))


############ MLP RECURSIVE     ##############################################

mlp_recursive_pred = mlp_recursive_method_pred(data2[:-HORIZON].filter(regex=str_filter),HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:].filter(regex=str_filter),prediction_reference_data.filter(regex=str_filter),mlp_recursive_pred,title="MLP recursive prediction",size=[2,2])
smape_arrays['smape_mlp_recursive'] = np.array(list(calculate_smape_df(prediction_reference_data.filter(regex=str_filter),mlp_recursive_pred).values()))


############ 1D CNN     #####################################################

cnn_pred = cnn_method_pred(data2[:-HORIZON].filter(regex=str_filter),HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:].filter(regex=str_filter),prediction_reference_data.filter(regex=str_filter),cnn_pred,title="CNN prediction",size=[2,2],plot_acf_pacf=False)
smape_arrays['smape_cnn'] = np.array(list(calculate_smape_df(prediction_reference_data.filter(regex=str_filter),cnn_pred).values()))

############ RNN VECTOR OUTPUT  #############################################

rnn_vector_output_pred = rnn_vector_method_pred(data2[:-HORIZON].filter(regex=str_filter),HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:].filter(regex=str_filter),prediction_reference_data.filter(regex=str_filter),rnn_vector_output_pred,title="RNN Vector Output prediction",size=[2,2],plot_acf_pacf=False)
smape_arrays['smape_rnn_vector'] = np.array(list(calculate_smape_df(prediction_reference_data.filter(regex=str_filter),rnn_vector_output_pred).values()))

############ RNN ENCODER DECODER  ###########################################

rnn_encoder_decoder_pred = rnn_enc_dec_method_pred(data2[:-HORIZON].filter(regex=str_filter),HORIZON,opt_lambdas,p_difference)
if show_plot:
    plot_predictions(training_data[-5*HORIZON:].filter(regex=str_filter),prediction_reference_data.filter(regex=str_filter),rnn_encoder_decoder_pred,title="RNN Encoder-Decoder prediction",size=[2,2],plot_acf_pacf=False)
smape_arrays['smape_rnn_encoder_decoder'] = np.array(list(calculate_smape_df(prediction_reference_data.filter(regex=str_filter),rnn_encoder_decoder_pred).values()))

#%% Extra plot
print("{:.2f} - smape_snaive_year".format(smape_arrays['smape_snaive_year'].mean()))
if show_plot:
    plot_predictions(training_data[-365:],prediction_reference_data,snaive_year_pred,title="Worst of Snaive year prediction",size=[10,2],selected_series=smape_arrays['smape_snaive_year'].argsort()[-20:][::-1])


#############################################################################
#############################################################################
###################### 6. GRID SEARCH BEST COMBO       ######################
#############################################################################
#############################################################################

#%%
#Print mean and plot smape values for each method:
fig, ax = plt.subplots(len(smape_arrays.keys()))
fig.suptitle("SMAPE values for all series per method")
for i,key in enumerate(smape_arrays.keys()):
    print(key+" "+str(smape_arrays[key].mean()))
    ax[i].plot(smape_arrays[key])
    ax[i].set_title(key)
    ax[i].set_ylim([0, 100])

#%%######### TEST RANDOM COMBINATIONS #######################################
results = {}
best = {'coeff':[],'smape':999}
for k in range(100):
    coeff = np.random.rand(2)
    results[k] = coeff
    multi_pred = (coeff[0]*snaive_decomp_pred+coeff[1]*snaive_year_pred)/sum(coeff)
    smape_multi = np.array(list(calculate_smape_df(prediction_reference_data,multi_pred).values())).mean()
    results[k+1000]=smape_multi
    if smape_multi<best['smape']:
        best['smape'] = smape_multi
        best['coeff'] = coeff
    print("{} - SMAPE: {}".format(coeff,smape_multi))

#%% FIND BEST ALGORITHM FOR EACH SERIES

best_algorithm_per_series = {}
new_smape_array = {}
#combined_predictions = pd.DataFrame(index=avg_pred.index)
for i,series_name in enumerate(data.filter(regex="^series").columns):
    best_algorithm_per_series[series_name] = list(smape_arrays.keys())[0]
    for k,algo in enumerate(smape_arrays.keys()):
        if smape_arrays[algo][i]<smape_arrays[best_algorithm_per_series[series_name]][i]:
            best_algorithm_per_series[series_name] = algo
    new_smape_array[series_name]=smape_arrays[best_algorithm_per_series[series_name]][i] 
    #combined_predictions[series_name]=globals()[best_algorithm_per_series[series_name][6:]+"_pred"][series_name]        
print("New overall smape: {}".format(np.array(list(new_smape_array.values())).mean()))
plt.hist(best_algorithm_per_series.values())


#############################################################################
#############################################################################
###################### 7. PREDICT FUTURE               ######################
#############################################################################
#############################################################################

#%%######### EMPTY               ############################################
avg_pred = avg_method_pred(data, HORIZON)

naive_pred = naive_method_pred(data, HORIZON)

snaive_pred = snaive_method_pred(data, HORIZON)

snaive_avg_pred = snaive_avg_method_pred(data, HORIZON, 35) #Best result for 35 days

snaive_exp_pred = snaive_exp_smoothing_method_pred(data, HORIZON,METHOD="holt",optimized=False,smoothing_slope=0.018,smoothing_level=.34)

snaive_median_pred = snaive_median_method_pred(data, HORIZON, 35) #Best result for 35 days

snaive_decomp_pred = snaive_decomp_method_pred(data2,HORIZON,opt_lambdas) 

combined_predictions = pd.DataFrame(index=avg_pred.index)
for i,series_name in enumerate(data.filter(regex="^series").columns):
    combined_predictions[series_name]=globals()[best_algorithm_per_series[series_name][6:]+"_pred"][series_name]        

mlp_multioutput_pred = mlp_multioutput_method_pred(data2.filter(regex="series-1"),HORIZON,opt_lambdas,p_difference)

mlp_recursive_pred = mlp_recursive_method_pred(data2,HORIZON,opt_lambdas,p_difference,True)

mlp_combination_pred = (mlp_recursive_pred + mlp_multioutput_pred)/2

rnn_encoder_decoder_pred = rnn_enc_dec_method_pred(data2,HORIZON,opt_lambdas,p_difference)

#############################################################################
#############################################################################
###################### 8. EXPORT RESULTS               ######################
#############################################################################
#############################################################################

#%%######### EXPORT PREDICTIONS TO KAGGLE CSV ###############################
                                                   
export_predictions_csv(fb_prophet_mod_pred,"fb_prophet_mod_pred.csv")

export_predictions_csv(snaive_avg_pred,"snaive_avg.csv")

export_predictions_csv(snaive_exp_pred,"snaive_exp.csv")

export_predictions_csv(multi_pred,"multi_combo.csv")

export_predictions_csv(combined_predictions,"multi_best_selected.csv")

export_predictions_csv(mlp_combination_pred,"mlp_combination_pred.csv")







