#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 09:46:18 2023

@author: yjy
"""

#%% 1.Load data

import os
import pandas as pd

work_dir = os.getcwd()
data = pd.read_excel(os.path.join(work_dir,"OUTPUT/df_output.xlsx"),index_col=(0))
df =data.iloc[:,1::2]
df.columns = data.iloc[:,::2].columns

# today's index
df_today = df.sort_index().tail(1).unstack()
df_today.reset_index(level = -1,drop = True, inplace = True )

# adjusted index (in 10 years from 2012-01-01)
df_selected = df.loc[df.index >= "2012-01-01"]
#df_range = df_selected.groupby(level = 0).apply(lambda x: max(x) - min(x))
#df_max = df_selected.unstack().groupby(level = 0).max()
#df_min = df_selected.unstack().groupby(level = 0).min()
#df_result = pd.concat([df_today,df_max,df_min],axis = 1)
#df_result.columns = ["today","max","min"]
#df_result["adj_today"] = (df_result["today"]- df_result["min"])/(df_result["max"]- df_result["min"])

df_adjusted = ((df_selected -df_selected.min())/(df_selected.max() - df_selected.min())).tail(1).T
df_result = pd.concat([df_today,df_adjusted],axis = 1)
df_result.reset_index(inplace = True)
df_result.columns = ["topic", "value","adj_value"]

df.sort_index().head(1)

#%% treemap
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

# create a treemap of the data using Plotly
fig = px.treemap(df_result, 
                 path=[px.Constant(''), 'topic'], 
                 values='value',
                 color='adj_value', 
                 #color_continuous_scale='RdBu_r',
                 color_continuous_scale='oranges',
                 hover_data={'value':':.2f', 'adj_value':':d'})

# show the treemap
#fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

fig.update_layout(font_size=20,font_family="Open Sans",font_color="#444")
fig.show()






