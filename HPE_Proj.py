# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:31:27 2021

@author: Abhiram
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

st.title('Ware Assistant')
today = datetime.date.today()
start_date = st.date_input('Input date', today)
start_date = pd.to_datetime(start_date)
start_date = start_date.isoformat()
st.write(start_date)