# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:27:11 2022

@author: Ted
"""
import pandas as pd

from Volta_model_4 import VoltaModel
import rbf_functions
#
# import time
# from datetime import timedelta

def volta_simulate(rbf_vars=0, data='historic',
                    energy_storage=0,
                    treatiesBenin = 0, treatiesBurkinaFaso =0,
                    treatiesCoteIvoire = 0, treatiesTogo=0,

                    Cjanuary=0,Cfebruary=0,Cmarch=0,
                    Capril=0, Cmay=0, Cjune=0,
                    Cjuly=0,Caugust=0, Cseptember=0,
                    Coctober=0,Cnovember=0, Cdecember=0,

                    waterUseBenin=0, waterUseBurkinaFaso=0,
                    waterUseCoteIvoire=0,waterUseTogo=0,

                    irriDemandMultiplier=1
                    ):
    
    n_inputs = 2  # (time, storage of Akosombo)
    n_outputs = 2 # Irrigation, Downstream:- (hydropower, environmental, floodcontrol)
    n_rbfs = n_inputs+2

    entry = rbf_functions.squared_exponential_rbf

    rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

    n_years = 29
    n_samples = 1

    
    l0_Akosombo = 241.0
    
    d0 = 505.0
    
    lowervolta_river = VoltaModel(l0_Akosombo, d0, n_years, n_samples, rbf, data=data,

                                  energy_storage = energy_storage,
                                  
                                  treatiesBenin = treatiesBenin, treatiesBurkinaFaso = treatiesBurkinaFaso,
                                  treatiesCoteIvoire = treatiesCoteIvoire, treatiesTogo= treatiesTogo,
                                  
                                  Cjanuary= Cjanuary,Cfebruary=Cfebruary,Cmarch=Cmarch,
                                  Capril=Capril, Cmay=Cmay, Cjune=Cjune,
                                  Cjuly=Cjuly,Caugust=Caugust, Cseptember=Cseptember,
                                  Coctober=Coctober,Cnovember=Cnovember, Cdecember=Cdecember,
                                  
                                  waterUseBenin=waterUseBenin, waterUseBurkinaFaso=waterUseBurkinaFaso, 
                                  waterUseCoteIvoire=waterUseCoteIvoire,waterUseTogo=waterUseTogo,
                                  
                                  irriDemandMultiplier=irriDemandMultiplier
                                  )
    
    lowervolta_river.set_log(True)
    output = lowervolta_river.evaluate(rbf_vars)
    
    return output #, lowervolta_river

###########Test function###########
df_rbf_vars = pd.read_csv("Savedsolutions/solution thining/filteredReleasePolicies.csv", header=None)
rbf_vars = df_rbf_vars.values[8]

output = volta_simulate(rbf_vars, data='historic')
print('done')
print(output)


