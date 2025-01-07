# -*- coding: utf-8 -*-

!pip install git+https://github.com/microsoft/dowhy.git
import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd
import graphviz
import networkx as nx
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)

# Import data from Github

url="https://raw.githubusercontent.com/sunnysong14/ContinualPerformanceValidityTSE2022/main/data/brackets.csv"

df = pd.read_csv(url, sep=",")

df

df.info()

df.describe().loc[['min', '25%', '50%', '75%', 'max']]

!apt install libgraphviz-dev
!pip install pygraphviz

data_df= df[[
'ndev',
'ns',
'nf',
'rexp',
'sexp',
'contains_bug',
'days_to_first_fix',
'exp']]

#--new graph----Final code
causal_graph = """strict digraph  {

sexp->ns;
sexp -> nuc;
nuc->nf;
ns->nf;
sexp->exp;
rexp->exp;
ndev->ns;
nf-> days_to_first_fix;
ndev->days_to_first_fix;
exp->days_to_first_fix;

}

"""

from dowhy import CausalModel
from IPython.display import Image, display

model= CausalModel(
        data = data_df,
        graph=causal_graph.replace("\n", " "),
        treatment='ndev',       #ndev
        outcome='days_to_first_fix')

#model.view_model()
model.view_model(file_name="causal_model.png") # Save the plot to a file
#display(Image(filename="causal_model.png"))

estimands = model.identify_effect()
print(estimands)

estimate= model.estimate_effect(
 identified_estimand=estimands,
 method_name='backdoor.linear_regression',
 confidence_intervals=True,
  test_significance=True
)

print(f'Estimate of causal effect: {estimate}')


estimate.interpret()

refutel_common_cause=model.refute_estimate(estimands,estimate,"random_common_cause")
print(refutel_common_cause)

refutel_common_cause=model.refute_estimate(estimands,estimate,"data_subset_refuter")
print(refutel_common_cause)

refutation = model.refute_estimate(estimands, estimate, method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=100)
print(refutation)

es_random=model.refute_estimate(estimands,estimate, method_name="random_common_cause", show_progress_bar= True)

print(es_random)

"""CATE"""

!pip install econml
import numpy as np
import pandas as pd
import logging

#import dowhy
#from dowhy import CausalModel
#import dowhy.datasets

import econml
import warnings
warnings.filterwarnings('ignore')

BETA = 10

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

dml_estimate = model.estimate_effect(estimands, method_name="backdoor.econml.dml.DML",
                                     control_value = 0, # Changed to 0 assuming days_to_first_fix is numeric
                                     treatment_value = 1, # Changed to 1 as a hypothetical treatment value.
                                 target_units = lambda df: df["rexp"]>1,  # condition used for CATE
                                 confidence_intervals=False,
                                method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                              'model_t': GradientBoostingRegressor(),
                                                              "model_final":LassoCV(fit_intercept=False),
                                                              'featurizer':PolynomialFeatures(degree=1, include_bias=False)},
                                               "fit_params":{}})
print(dml_estimate)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from econml.inference import BootstrapInference
dml_estimate = model.estimate_effect(estimands,
                                     method_name="backdoor.econml.dml.DML",
                                     target_units = "ate",
                                     confidence_intervals=True,
                                     method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                              'model_t': GradientBoostingRegressor(),
                                                              "model_final": LassoCV(fit_intercept=False),
                                                              'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                                               "fit_params":{
                                                               'inference': BootstrapInference(n_bootstrap_samples=100, n_jobs=-1),
                                                            }
                                              })
print(dml_estimate)
