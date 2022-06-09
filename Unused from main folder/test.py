import os
import numpy as np

import rbf_functions
#Install using pip and restart kernel if unavailable
import utils #install using:  pip install utils
from numba import njit #pip install numba #restart kernel (Ctrl + .)
import time
import pandas as pd
from datetime import timedelta

start = time.time()

# def create_path(rest):
#     my_dir = os.path.dirname(os.path.realpath(__file__))
#     return os.path.abspath(os.path.join(my_dir, rest))
#
#
data = '1000x1'
n_samples= 1000
n_years = 1
n_days_one_year = 365
#
# evap_Ak = utils.loadArrangeMatrix(
#     create_path("./Data/Stochastic/evap_Ak-"+ data +".csv"),
#     n_samples, n_days_one_year * n_years
#     ) #evaporation losses @ Akosombo_ stochastic data (inches per day)
# inflow_Ak = utils.loadArrangeMatrix(
#     create_path("./Data/Stochastic/InflowAk-"+ data +".csv"),
#     n_samples, n_days_one_year * n_years
# ) * 35.3146  # inflow, i.e. flows to Akosombo_stochastic data
# tailwater_Ak = utils.loadArrangeMatrix(
#     create_path("./Data/Stochastic/tailwater_Ak-"+ data +".csv"),
#     n_samples, n_days_one_year * n_years
#     ) # tailwater level @ Akosombo (ft)
# fh_Kpong = utils.loadArrangeMatrix(
#     create_path("./Data/Stochastic/fhKp-"+ data +".csv"),
#     n_samples, n_days_one_year * n_years
#     )

evap_Ak = np.reshape(pd.read_csv("./Data/Stochastic/evap_Ak-"+ data +".csv", header=None).to_numpy(), (n_samples, n_days_one_year * n_years))
#evaporation losses @ Akosombo_ stochastic data (inches per day)

inflow_Ak = np.reshape(pd.read_csv("./Data/Stochastic/InflowAk-"+ data +".csv", header=None).to_numpy(), (n_samples, n_days_one_year * n_years)) * 35.3146
#inflow, i.e. flows to Akosombo_stochastic data

tailwater_Ak = np.reshape(pd.read_csv("./Data/Stochastic/tailwater_Ak-"+ data +".csv", header=None).to_numpy(), (n_samples, n_days_one_year * n_years))
#tailwater level @ Akosombo (ft)

fh_Kpong = np.reshape(pd.read_csv("./Data/Stochastic/fhKp-"+ data +".csv", header=None).to_numpy(), (n_samples, n_days_one_year * n_years))


print('done')
end = time.time()
print("The time of execution of above program is :", str(timedelta(seconds=end-start)))