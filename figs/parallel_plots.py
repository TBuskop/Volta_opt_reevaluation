# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:07:14 2022

@author: Ted
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from ema_workbench.analysis import parcoords
import seaborn as sns

#Set the plotsizes
plt.rcParams['figure.figsize'] = [20, 10] # set figure size

#Read in data
data = pd.read_csv('Saved solutions/hydro_max2.csv')

#Create normalised data
data_no_name = data.drop('Name', axis=1,)
x_data = data_no_name.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_data_scaled = min_max_scaler.fit_transform(x_data)
data_norm = pd.DataFrame(x_data_scaled)

#Add titles to normalised columns 
data_norm.columns = data.drop('Name', axis=1).columns

#Plot
limits = parcoords.get_limits(data_norm)
axes = parcoords.ParallelAxes(limits)

axes.plot(data_norm)


#Plot function that plots figure with legend.
def axes_paraplot(outcomes_normalised, outcomes_with_policy_names, save_name_figure):
    
    policy_names = outcomes_with_policy_names['Name'].values # for legend
    
    limits = parcoords.get_limits(outcomes_normalised)
    axes = parcoords.ParallelAxes(limits)

    for i in range(len(outcomes_normalised)):   
        axes.plot(outcomes_normalised.iloc[i], label=policy_names[i], color=sns.color_palette('hls', len(outcomes_normalised))[i])
    axes.legend()
    
   #Uncomment if you want to save the plots 
   # plt.savefig(f'../plots/{save_name_figure}.png', bbox_inches="tight")
    
axes_paraplot(data_norm, data, 'test')