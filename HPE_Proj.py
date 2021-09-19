# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:31:27 2021

@author: Abhiram
"""
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
plt.style.use('default')
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.datasets import mnist
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
from statsmodels.tools.eval_measures import meanabs
from dateutil.relativedelta import relativedelta
import datetime

st.title('Ware Assistant')
today = datetime.date.today()
start_date = st.date_input('Input date', today)
start_date = pd.to_datetime(start_date)
#start_date = start_date.isoformat()
#st.write(start_date)
st.write("Preparing data...........")
df = pd.read_excel('data_monthly.xlsx')
df['Sale Date'] = pd.to_datetime(df['Sale Date'])
df.set_index('Sale Date', inplace=True)
df = df.resample('MS').sum()
df = df.reset_index()
df.rename(columns = {'Model':'sales'},inplace=True)
df.to_csv('cardealer.csv')

dpred = start_date#pd.to_datetime((input("Date Input")))

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month
st.write("Spiltting the dataset......................")
n = diff_month(dpred , df.iloc[len(df)-1,0])

for _ in range(4):## Main code
    d = df.iloc[len(df)-1 ,0]
    future_date = d + relativedelta(months=1)
    df.loc[len(df.index)] = [future_date,0]
    df_diff = df.copy()#add previous sales to the next row
    df_diff['prev_sales'] = df_diff['sales'].shift(1)#drop the null values and calculate the difference
    df_diff = df_diff.dropna()
    df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])
    df_supervised = df_diff.drop(['prev_sales'],axis=1)#adding lags
    for inc in range(1,13):
        field_name = 'lag_' + str(inc)
        df_supervised[field_name] = df_supervised['diff'].shift(inc)#drop null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)
    m = -12
    df_model = df_supervised.drop(['sales','Sale Date'],axis=1)#split train and test set
    train_set, test_set = df_model[:m].values, df_model[m:].values
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)# reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    st.write("Running LSTM....................")
    regressor = Sequential()
    regressor.add(LSTM(units = 5, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.6))
    #regressor.add(LSTM(units = 1, return_sequences = True))
    #regressor.add(Dropout(0.9))
    regressor.add(Dense(units = 1))
    regressor.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error'])
    history = regressor.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=100, batch_size=1, verbose=1, shuffle=False)
    y_pred = regressor.predict(X_test,batch_size=1)#for multistep prediction, you need to replace X_test values with the predictions coming from t-1
    y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])#rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        print(np.concatenate([y_pred[index],X_test[index]],axis=1))
        pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))#reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])#inverse transform
    pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
    result_list = []
    sales_dates = list(df[m-1:]['Sale Date'])
    act_sales = list(df[m-1:].sales)
    for index in range(0,len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_dict['Sale Date'] = sales_dates[index+1]
        result_list.append(result_dict)
    df_result = pd.DataFrame(result_list)
    df_sales_pred = pd.merge(df,df_result,on='Sale Date',how='left')
    df.iloc[len(df)-1,1] = df_sales_pred.iloc[len(df)-1,2] # replacing
output = df_sales_pred.iloc[len(df_sales_pred)-1,2]
st.header('Predicted Sales:')
st.write(output)
