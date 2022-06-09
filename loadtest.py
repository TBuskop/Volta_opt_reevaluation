from ema_workbench import load_results
from ema_workbench.analysis import parcoords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "C:/Users/busko/OneDrive - Delft University of Technology/TU Delft/Engineering Policy Analysis (EPA)/Year 2/Afstuderen/Results cloud services/results1.tar.gz"
experiments, outcomes = load_results(path)