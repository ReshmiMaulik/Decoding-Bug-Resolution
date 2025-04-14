# -*- coding: utf-8 -*-
"""CauSE-Causal Inference.ipynb
"""

#!pip install 'scipy==1.10.1'
#!pip install 'scikit-learn==1.2.2'
#!pip install --force-reinstall econml

!pip install git+https://github.com/microsoft/dowhy.git
import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd

# Import data from Github

#url="https://raw.githubusercontent.com/sunnysong14/ContinualPerformanceValidityTSE2022/main/data/vscode.csv"
#url="https://raw.githubusercontent.com/sunnysong14/ContinualPerformanceValidityTSE2022/main/data/tensorflow.csv"

# Data files are uploaded
from google.colab import files
uploaded = files.upload()

# Read Data file
df = pd.read_csv('vscode.csv')
#df = pd.read_csv('tensorflow.csv')
#df = pd.read_csv(url, sep=",")


print(df.shape)

print(df.columns)

df = df[['ns','nf','ndev','exp','rexp','sexp','nuc','days_to_first_fix','entropy']]

print(df.shape)
df.head()

# To see the correlation among the data variables
import matplotlib.pyplot as plt
import seaborn as sns
# plot heatmap for feature variables
corr = df.corr()
plt.figure(figsize=[12,10])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,annot_kws={"size":15})
#plt.xticks(rotation=60)
import matplotlib.pyplot as plt
import seaborn as sns
# plot heatmap for feature variables
corr = df.corr()
plt.figure(figsize=[12,10])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,annot_kws={"size":15})
#plt.xticks(rotation=60)
#plt.title("Heatmap of Correlation Coefficient for Bug Feature Variables", size=10);
plt.xticks(rotation=45,size=15)
plt.yticks(rotation=0, size=15)
#plt.title("Heatmap of Correlation Coefficient for Bug Feature Variables", size=14);

plt.savefig('corrplot.png', bbox_inches='tight', pad_inches=0.0)

"""Since here entrophy and nuc has high correlation and sexp has high correlation with exp so we drop the columns"""

import graphviz
!apt install libgraphviz-dev
!pip install pygraphviz

#--new graph----DAG code
causal_graph = """strict digraph  {

sexp->ns;

ns->nf;
sexp->exp;
rexp->exp;
ndev->ns;
nf-> days_to_first_fix;
ndev->days_to_first_fix;
exp->days_to_first_fix;
entropy->ns;
entropy->nf;

}

"""

"""Here, we have implicitly defined a causal graph by setting the type of treatment and the number of common causes. DoWhy stores graph objects in the DOT language, which gives us a convenient way of specifying our own directed causal graph (digraph) once we’re working with real-world data.

Finally, we combine all of this information into one single causal model.
"""

#'sexp', 'rexp','ndev','entropy'
# With graph
model=CausalModel(
        data = df,
        treatment='ndev',
        outcome='days_to_first_fix',
        graph=causal_graph,  # Pass the causal_graph variable directly
      #  instruments=['ndev']
        )
model.view_model(layout="dot")
model.view_model(file_name="causal_model.png") # Save the plot to a file


# To contruct the SCM

#Model
import networkx as nx
causal_graph1 = nx.DiGraph([('sexp', 'ns'), ('ns', 'nf'), ('sexp', 'exp'),('rexp','exp'), ( 'exp','days_to_first_fix'), ('nf', 'days_to_first_fix'),('ndev', 'ns'),('exp', 'days_to_first_fix'),('entropy', 'ns'), ('entropy','nf')])

from dowhy import CausalModel, gcm
causal_model = gcm.StructuralCausalModel(causal_graph1)

# Set causal mechanisms for each node
causal_model.set_causal_mechanism('sexp', gcm.EmpiricalDistribution())
causal_model.set_causal_mechanism('ns', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('nf', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('exp', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('rexp', gcm.EmpiricalDistribution())
causal_model.set_causal_mechanism('days_to_first_fix', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
causal_model.set_causal_mechanism('ndev', gcm.EmpiricalDistribution())

causal_model.set_causal_mechanism('entropy', gcm.EmpiricalDistribution())

# Now fit the model , Fitting the SCM to the data
gcm.fit(causal_model, df)
#Fitting means, we learn the generative models of the variables in the SCM according to the data.
#Once fitted, we can also obtain more insights into the model performances:

print(gcm.evaluate_causal_model(causal_model, df))


"""Identification-The identification step involves defining what to measure by analyzing the causal graph. However, the actual evaluation of identification utilizes the available data and is performed during the estimation step.

DoWhy offers a range of algorithms that can be used to determine if a desired causal effect can be identified when given a particular causal model.
"""

## Check whether causal effect is identified and return target estimands
identified_estimand = model.identify_effect()
print(identified_estimand)

"""Estimate The causal effect of the treatments on the outcome is determined by the change in the value of the treatment variable, and we evaluate the strength of the effect by statistical estimation. There are many ways to estimate the causal effect, but in our demo, we will stick to linear regression.

Next, we can focus on estimation, which is the process of quantifying the target effect using the available data.
"""

estimate= model.estimate_effect(
 identified_estimand,
 method_name='backdoor.linear_regression',
 confidence_intervals=True,
  test_significance=True
)

print(f'Estimate of causal effect: {estimate}')

# Textual Interpreter
interpretation = estimate.interpret(method_name="textual_effect_interpreter")

"""Refute the estimate

The causal effect estimate is subjected to refutation tests to assess its robustness to unverified assumptions. The refutation process involves incorporating changes such as adding random variables or observed common causes to the causal model.
"""

refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
print(refute_results)

refutel_common_cause=model.refute_estimate(identified_estimand,estimate,"data_subset_refuter")
print(refutel_common_cause)

refutel_common_cause

"""Checking if the estimate is correct"""

print("DoWhy estimate is " + str(estimate.value))
#print ("Actual true causal effect was {0}".format(rvar))
