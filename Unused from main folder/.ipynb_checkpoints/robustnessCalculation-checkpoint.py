# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:23:37 2022

@author: Ted
"""

from ema_workbench import load_results

path = r'./reevaluation/results.tar.gz'
experiments, outcomes = load_results(path)


# robustnesMetricsUsed = [robustness_metrics.maximin, 
#                         robustness_metrics.minimax_regret90, 
#                         robustness_metrics.mean_variance, 
#                         robustness_metrics.percentile_based_peakedness]

# for metric in robustnesMetricsUsed:
#     overall_scores = {}
#     for policy in np.unique(experiments['policy']):
#         scores = {}
        
#         toAssess = experiments['policy']==policy
        
#         for outcome in model.outcomes:
#             value  = outcomes[outcome.name][toAssess]
#             metric_score = metric(value, outcome.kind)
#             scores[outcome.name] = metric_score
#         overall_scores[policy] = scores
#     scores = pd.DataFrame.from_dict(overall_scores).T