#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Here itself we dont using pickle.load because of github.
# Load model data
model_data = joblib.load('Apple_project.pkl')

params = model_data['params']
order = model_data['order']
seasonal_order = model_data['seasonal_order']


# In[3]:


# Load dataset
df = pd.read_csv('P675 DATASET.csv', index_col='Date', parse_dates=True)

# Reset index for SARIMAX forecasting
close_data = df['Close'].reset_index(drop=True)

model = SARIMAX(
    close_data,
    order=order,
    seasonal_order=seasonal_order
)
final_fit = model.filter(params)


# In[4]:


st.title('Apple Stock Price Prediction')


# In[5]:


def user_input_parameter():

    forecast_days = st.sidebar.slider("Select Forecast Days",min_value=1,max_value=30,value=7)
    data = {'Forecast_Days': forecast_days}
    return forecast_days

forecast_days=user_input_parameter()
fut_pred=final_fit.forecast(steps=forecast_days)
forecast_df = fut_pred.to_frame(name="Forecasted Price")

st.subheader("Future Stock Price Forecast")
st.dataframe(fut_pred)
st.subheader('Forecast Graph')
st.line_chart(fut_pred)

