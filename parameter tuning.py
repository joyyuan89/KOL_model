#%% Setup
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:42:20 2023

@author: jakewen
"""

# libraries
import pandas as pd
import numpy as np
import os

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/df_output.xlsx"
df_output = pd.read_excel(input_path, index_col=0)

#%% Load market data
input_path_market_data = "/Users/jakewen/Desktop/Github/KOL_model/^VIX.csv"
VIX_data = pd.read_csv(input_path_market_data, index_col=0)
VIX_data.index = pd.to_datetime(VIX_data.index)

#%% Merge data

df_merged = pd.merge(df_output, VIX_data, left_index=True, right_index=True, how='inner')

#%% Plot
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)

fig,ax_1 = plt.subplots()
ax_2 = ax_1.twinx()

x = df_merged.index
y = df_merged["value"]
z = df_merged["Close"]

ax_1.plot(x,y,label='Crisis level')
ax_2.plot(x,z,color="orange", label = 'VIX')

ax_1.legend(loc='upper left', fontsize=20)
ax_2.legend(loc='upper right', fontsize=20)

plt.show()
