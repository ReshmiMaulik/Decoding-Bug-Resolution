The datasets utilised for this study comprise 13 datasets, stored on GitHub and accessible via https://raw.githubusercontent.com/sunnysong14/ContinualPerformanceValidityTSE2022/main/data/brackets.csv. 

This notebook explores causal inference and interpretability techniques to understand relationships among various factors, such as the number of developers, their experience, and changes in the code, with a focus on how different variables affect 'ET' (bug resolution time).

The steps followed are:

1. Setup and Data Loading.
2. DAG build-up using domain knowledge.
3. Structural Causal Model (SCM).
4. Determine the estimand using the model.
5. Causal estimation using the backdoor regression.
6. Refutation Tests.
7. Meta-analyses with forest plots are used to summarise causal effects across different projects, enabling us to identify
global patterns.


We have used the following version:
### Python Version
3.12.13 [GCC 11.4.0]

### DoWhy Version
0.0.0

### Dependency Versions
pandas version: 2.2.2
numpy version: 2.0.2
scikit-learn version: 1.6.1
scipy version: 1.16.3
statsmodels version: 0.14.6
networkx version: 3.6.1
