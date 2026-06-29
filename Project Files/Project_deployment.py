#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


final_model=pickle.load(open('Apple_project.pkl','rb'))


# In[3]:


st.title('Apple Stock Price Prediction')


# In[4]:


def user_input_parameter():

    forecast_days = st.sidebar.slider("Select Forecast Days",min_value=1,max_value=30,value=7)
    data = {'Forecast_Days': forecast_days}
    return forecast_days

forecast_days=user_input_parameter()
final_fit=final_model.fit()
fut_pred=final_fit.forecast(steps=forecast_days)
forecast_df = fut_pred.to_frame(name="Forecasted Price")

st.subheader("Future Stock Price Forecast")
st.dataframe(fut_pred)
st.subheader('Forecast Graph')
st.line_chart(fut_pred)

