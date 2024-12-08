MANUSCRIPT: Different environmental factors predict the occurrence of Tick-Borne Encephalitis virus (TBEV) and reveal new potential risk areas across Europe via geospatial models. 

This repository contains the structure and components utilized in producing the results indicated in the manuscript. It contains solely code related to dependent and independent variable pre-processing, model-specific pre-processing, and model parameters and methodology. Most raw information is contained in the Appendix of the manuscript, so we omit any raw or intermediate data files needed to run the processes; code is included mainly for informative purposes. 

*****************************

REPOSITORY STRUCTURE
src/
│
├── config/
│   └── Configuration files and settings for the project.
│
├── utils/
│   └── Utility functions and helper scripts for data processing, feature engineering, etc.
│
├── notebooks/
│   └── Jupyter notebooks for exploratory data analysis (EDA), prototyping, and model experimentation.

*****************************

USAGE

CONFIG
All configurable settings for the project (e.g., model hyperparameters, file paths, data settings) are stored in the src/config folder. Update these files as needed for specific experiments.

UTILITIES
Reusable scripts and helper functions are located in the src/utils folder. Examples include:

Modeling functions
Plotting functions
Processing helper functions

NOTEBOOKS
The src/notebooks folder contains the core source code for the entirety of the analysis, starting from raw data ingestion to model selection and evaluation. The notebooks are numbered from 1.0 to 3.2, with each whole integer indicating a step in the process
1:  Raw focus data pre-processing, including k-means clustering to produce the three distinct focus regions. 
2.1 - 2.4: Landscape, climate, and biological / host covariate processing. Some image processing was accomplished using QGIS, which omitted here with discussion in the manuscript. 
3.0 - 3.2: Model-specific methods, from preprocessing to model development. The maxent model was developed using Maxent version 3.4.4.
*****************************

Prerequisites
Python 3.x
Necessary libraries (e.g., numpy, pandas, scikit-learn, matplotlib, etc.). See the requirements.txt file for the full list of dependencies.
