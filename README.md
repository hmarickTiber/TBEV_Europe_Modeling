
---

# Usage

### Configuration (`src/config/`)

All project-specific settings (such as model hyperparameters, file paths, and data configurations) are stored in the `src/config` folder. Update these files as necessary to customize for your specific experiments.

### Utilities (`src/utils/`)

Reusable utility scripts and helper functions for various tasks are located in the `src/utils/` folder. Key examples include:

- **Modeling Functions:** Code to build and evaluate models
- **Plotting Functions:** Visualization code for results
- **Processing Functions:** Preprocessing and feature engineering helpers

### Notebooks (`src/notebooks/`)

The `src/notebooks/` folder contains the primary source code for the analysis pipeline. The notebooks are organized in a step-by-step manner, from raw data ingestion to model selection and evaluation:

- **Notebook 1:** Raw data preprocessing, including k-means clustering to define the three focus regions.
- **Notebooks 2.1 - 2.4:** Landscape, climate, and biological/host covariate processing. Note: Some image processing was done using QGIS and is omitted here, but discussed in the manuscript.
- **Notebooks 3.0 - 3.2:** Model-specific methods, from preprocessing to model development. The Maxent model was built using version 3.4.4.

---

# Prerequisites

- **Python 3.x**
- **Required Libraries:**  
  The following libraries are needed to run the code. The full list of dependencies can be found in the `requirements.txt` file:

  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - *(and others as per `requirements.txt`)*

