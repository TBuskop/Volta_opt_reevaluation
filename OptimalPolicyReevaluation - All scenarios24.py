# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:11:17 2022

@author: Ted
"""

import pandas as pd
import time
from datetime import timedelta

from ema_workbench import (Model, RealParameter, Constant,
                           MultiprocessingEvaluator,
                           save_results, ema_logging, ArrayOutcome, Scenario)

from Volta_model_4_function import volta_simulate

optimisedPolicies = pd.read_csv("Savedsolutions/solution thining/filteredReleasePolicies.csv", header=None)
n_releasePol = len(optimisedPolicies)
n_data_sets = ['50x20'] # could add '1000x1'
model_set = []

for j in range(len(n_data_sets)):

    for i in range(n_releasePol):
        model = Model(f'ReleasePol{i}Set{n_data_sets[j]}', function=volta_simulate)


        # model.levers = [RealParameter("energy_storage", 0, 5000), #implemented
        #                 RealParameter("treatiesBenin", 0, 1), RealParameter("treatiesBurkinaFaso", 0, 1), #implemented
        #                 RealParameter("treatiesCoteIvoire", 0, 1), RealParameter("treatiesTogo", 0, 1)
        #                 ]

        #specify uncertainties
        model.uncertainties = [RealParameter('Cjanuary', -0.1, 1), RealParameter('Cfebruary', -0.25, 0.03),  #implemented
                               RealParameter('Cmarch', -0.25, 0.03), RealParameter('Capril', -0.25, 0.03),
                               RealParameter('Cmay', -0.1, 0.2), RealParameter('Cjune', 0.1, 0.55),
                               RealParameter('Cjuly', -0.15, -0.1), RealParameter('Caugust', -0.08, 0.25),
                               RealParameter('Cseptember', 0.25, 0.40), RealParameter('Coctober', 0.18, 0.45),
                               RealParameter('Cnovember', 0.05, 0.20), RealParameter('Cdecember', -0.75, 0),

                               RealParameter('waterUseBenin', 0, 1), RealParameter('waterUseBurkinaFaso', 0, 1),  #implemented
                               RealParameter('waterUseCoteIvoire', 0, 1), RealParameter('waterUseTogo', 0, 1),

                               RealParameter('irriDemandMultiplier', 0.5, 3),  # implemented

                               RealParameter("energy_storage", 0, 5000),  # implemented
                               RealParameter("treatiesBenin", 0, 1), RealParameter("treatiesBurkinaFaso", 0, 1),# implemented
                               RealParameter("treatiesCoteIvoire", 0, 1), RealParameter("treatiesTogo", 0, 1)
                               ]

        #specify outcomes
        # note how we need to explicitely indicate the direction
        model.outcomes = [ArrayOutcome('j_hyd_a'),
                          ArrayOutcome('j_hyd_kp'),
                          ArrayOutcome('j_energy_reliability'),
                          ArrayOutcome('j_irri'),
                          ArrayOutcome('j_env'),
                          ArrayOutcome('j_fldcntrl'),
                          ]

        rbf_vars = optimisedPolicies.values[i]

        model.constants = [Constant('rbf_vars', rbf_vars),
                           Constant('data', n_data_sets[j])]

        model_set.append(model)


ema_logging.log_to_stderr(ema_logging.INFO)

uncertainty_scenarios = pd.read_csv('./reevaluation_results/uncertainty_designs.csv', index_col=0)
experiments_slice1 = uncertainty_scenarios[0:int(2400/2)]
experiments_slice2 = uncertainty_scenarios[1200:1600]
experiments_slice22 = uncertainty_scenarios[1600:2000]
experiments_slice23 = uncertainty_scenarios[2000:2200]
experiments_slice24 = uncertainty_scenarios[2200:2400]

scenarios1 = [Scenario(f"{index}", **row) for index, row in experiments_slice1.iterrows()]
scenarios2 = [Scenario(f"{index}", **row) for index, row in experiments_slice2.iterrows()]
scenarios22 = [Scenario(f"{index}", **row) for index, row in experiments_slice22.iterrows()]
scenarios23 = [Scenario(f"{index}", **row) for index, row in experiments_slice23.iterrows()]
scenarios24 = [Scenario(f"{index}", **row) for index, row in experiments_slice24.iterrows()]

# n_scenarios = 2400
n_policies = 0


def main():
    start = time.time()
    with MultiprocessingEvaluator(model_set) as evaluator:
        results = evaluator.perform_experiments(scenarios24, n_policies)
        save_results(results, r'./reevaluation_results/results24.tar.gz')
    end = time.time()
    print("The time of execution of above program is :", str(timedelta(seconds=end-start)))

if __name__ == "__main__":
    main()


    
