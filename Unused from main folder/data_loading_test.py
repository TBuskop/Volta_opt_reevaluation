# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:51:02 2022

@author: Ted
"""
import os
import numpy as np

import rbf_functions 
#Install using pip and restart kernel if unavailable
import utils #install using:  pip install utils
from numba import njit #pip install numba #restart kernel (Ctrl + .)
import pandas as pd

# Original code ----------------------------
n_days_one_year = 365
n_years = 2

# define path
def create_path(rest): 
    my_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(my_dir, rest))

tailwater_Ak = utils.loadArrangeMatrix(
    create_path("./Data/Historical_data/matrix/tailwaterAk_history.txt"), 
    n_years, 
    n_days_one_year
    )
#---------------------------------------------

# New code ----------------------------
datadf = pd.read_csv('./Data/Historical_data/matrix/tailwaterAk_history.txt', sep=" ", header=None)
data = datadf.to_numpy()[0]

data_cutoff = data[0: n_years * n_days_one_year]
data_reshaped = data_cutoff.reshape(n_years, n_days_one_year)

#----------------------------------------------------


print("data as loaded in model (values missing in between):", tailwater_Ak, "\n")
print("Amount of NaN values in set: ", datadf.isna().sum().sum(), "\n")

print("old data shape: ", data.shape)



print("new data shape: ", data_reshaped.shape)