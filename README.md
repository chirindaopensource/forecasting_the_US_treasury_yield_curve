# **`README.md`**

# Forecasting the U.S. Treasury Yield Curve: A Distributionally Robust Machine Learning Approach

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.04608-b31b1b.svg)](https://arxiv.org/abs/2601.04608)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2601.04608)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Discipline](https://img.shields.io/badge/Discipline-Computational%20Finance%20%7C%20Operations%20Research-00529B)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Data Sources](https://img.shields.io/badge/Data-LSEG%20Reuters%20Workspace-lightgrey)](https://www.lseg.com/en/data-analytics)
[![Core Method](https://img.shields.io/badge/Method-Distributionally%20Robust%20Optimization-orange)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Analysis](https://img.shields.io/badge/Analysis-Factor--Augmented%20Dynamic%20Nelson--Siegel-red)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Validation](https://img.shields.io/badge/Validation-Random%20Forest%20Ensemble-green)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Robustness](https://img.shields.io/badge/Robustness-Adaptive%20Forecast%20Combination-yellow)](https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-blue.svg)](https://www.statsmodels.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-ff69b4)](https://shap.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Forecasting the U.S. Treasury Yield Curve: A Distributionally Robust Machine Learning Approach"** by:

*   **Jinjun Liu** (Hong Kong Baptist University)
*   **Ming-Yen Cheng** (Hong Kong Baptist University)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and cleansing of zero-coupon yields and macroeconomic indicators to the rigorous estimation of Factor-Augmented Dynamic Nelson-Siegel (FADNS) models and high-dimensional Random Forests, culminating in distributionally robust forecast combinations that minimize worst-case expected loss under ambiguity.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_main_study_pipeline_variant`](#key-callable-run_main_study_pipeline_variant)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Liu and Cheng (2026). The core of this repository is the iPython Notebook `forecasting_the_US_treasury_yield_curve_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline recasts yield curve forecasting as an operations research problem, where the objective is to select a decision rule that minimizes worst-case expected loss over an ambiguity set of forecast error distributions.

The paper addresses the challenge of forecasting U.S. Treasury yields in an environment of distributional uncertainty and structural instability. This codebase operationalizes the paper's framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Cleanse and normalize zero-coupon yields and high-dimensional macroeconomic panels, enforcing strict publication lags to prevent look-ahead bias.
-   Estimate rolling-window Factor-Augmented Dynamic Nelson-Siegel (FADNS) models using principal components extracted from economic indicators.
-   Train high-dimensional Random Forest models to capture nonlinear interactions among macro-financial drivers.
-   Implement distributionally robust forecast combination schemes (DRO-ES, DRMV) that penalize downside tail risk and stabilize covariance estimation.
-   Evaluate performance using rigorous metrics such as Root Mean Squared Forecast Error (RMSFE) across maturities and horizons.

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Machine Learning, and Robust Optimization.

**1. Factor-Augmented Dynamic Nelson-Siegel (FADNS):**
The yield curve is modeled using the dynamic Nelson-Siegel framework, augmented with latent factors extracted from a large panel of macroeconomic variables via Principal Component Analysis (PCA).
-   **Measurement Equation:** $y_t(\tau) = L(\tau) \beta_t + \varepsilon_t$
-   **State Dynamics:** $X_{t+1}^{(k)} = c^{(k)} + \Phi^{(k)} X_t^{(k)} + \eta_t$, where $X_t^{(k)} = [\beta_t, F_t^{(k)}]$.

**2. Nonparametric Machine Learning (Random Forest):**
To capture nonlinearities, Random Forest models are trained on a high-dimensional feature set including lagged macro indicators and lagged yields.
-   **Forecasting:** $\hat{y}_{t+h|t} = \hat{g}_{h,\tau}(W_t)$, where $W_t$ includes lagged predictors.
-   **Optimization:** Hyperparameters are tuned via Randomized Cross-Validation respecting the time-series structure.

**3. Distributionally Robust Optimization (DRO):**
Forecast combination weights are optimized to minimize worst-case expected loss over an ambiguity set, rather than assuming a fixed error distribution.
-   **DRO-ES:** Weights are exponentially reweighted based on Expected Shortfall (ES) loss: $w_k \propto \exp(\eta \text{ES}_\alpha(e_k))$.
-   **Regularized Mean-Variance:** Weights are derived from a ridge-regularized covariance matrix to handle estimation uncertainty.

**4. Adaptive Forecast Combination (AFTER):**
Weights are updated recursively based on past performance, allowing the ensemble to adapt quickly to regime shifts.

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve/blob/main/forecasting_the_US_treasury_yield_curve_summary_two.png" alt="Forecasting Framework Summary" width="100%">
</div>

## Features

The provided iPython Notebook (`forecasting_the_US_treasury_yield_curve_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 40 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (rolling window sizes, lag structures, optimization penalties) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, unit consistency, and temporal alignment.
-   **Deterministic Execution:** Enforces reproducibility through seed control, deterministic sorting, and rigorous logging of all stochastic outputs.
-   **Comprehensive Evaluation:** Computes RMSFE tables across 15 maturities and 5 horizons, along with robustness checks for benchmark yields and global markets.
-   **Reproducible Artifacts:** Generates structured dictionaries, serializable outputs, and cryptographic manifests for every intermediate result.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-6):** Ingests raw yields and macro data, validates schemas, aligns indices, cleanses missing values, and interpolates quarterly series.
2.  **Structural Break Detection (Tasks 7-8):** Applies CUSUM and PELT algorithms to identify regime shifts in the yield curve.
3.  **DNS Modeling (Tasks 9-13):** Constructs Nelson-Siegel loadings, extracts latent factors via cross-sectional regression, estimates rolling VAR dynamics, and generates parametric forecasts.
4.  **FADNS Modeling (Tasks 14-20):** Preprocesses macro data (stationarity, standardization), extracts PCA factors, estimates augmented VAR models, and selects the optimal factor dimension $k^*$.
5.  **Random Forest Modeling (Tasks 21-25):** Constructs high-dimensional feature sets, normalizes data, trains RF models with randomized CV, and generates nonparametric forecasts.
6.  **Forecast Combination (Tasks 26-28):** Builds forecast pools, computes combination weights using 14 different schemes (Classic, Var/Risk, DRO, AFTER), and evaluates combined performance.
7.  **Robustness & Extensions (Tasks 30-35):** Performs robustness checks using benchmark coupon-bearing yields, TIC variable augmentation, and extends the analysis to global sovereign bond markets.
8.  **Interpretability & Visualization (Tasks 36-39):** Computes SHAP values for model interpretation and prepares data for visualizing weight dynamics and error dynamics.
9.  **Packaging (Task 40):** Aggregates all results into a final artifact bundle.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 40 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_main_study_pipeline_variant`

The project is designed around a single, top-level user-facing interface function:

-   **`run_main_study_pipeline_variant`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between cleansing, modeling, combination, and evaluation modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `shap`, `ruptures`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve.git
    cd forecasting_the_US_treasury_yield_curve
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scikit-learn scipy statsmodels shap ruptures pyyaml
    ```

## Input Data Structure

The pipeline requires six primary DataFrames (mocked in the example usage):
1.  **`df_us_yields_raw`**: U.S. zero-coupon equivalent yields (15 maturities).
2.  **`df_us_macro_raw`**: U.S. macroeconomic indicators (111 variables).
3.  **`df_us_benchmark_yields_raw`**: U.S. benchmark coupon-bearing yields (9 maturities).
4.  **`df_us_tic_raw`**: Treasury International Capital (TIC) flow variables.
5.  **`df_global_yields_raw`**: Global 10-year sovereign bond yields (7 countries).
6.  **`global_macro_panels_raw`**: Dictionary of macroeconomic panels for global countries.

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `run_main_study_pipeline_variant` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    # (Simulated in the notebook example)
    raw_study_config = STUDY_CONFIG 
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    (
        df_us_yields_raw, 
        df_us_macro_raw, 
        df_us_benchmark_yields_raw, 
        df_us_tic_raw, 
        df_global_yields_raw, 
        global_macro_panels_raw, 
        study_metadata
    ) = generate_synthetic_data()

    # 3. Execute the entire replication study.
    artifacts = run_main_study_pipeline_variant(
        df_us_yields_raw=df_us_yields_raw,
        df_us_macro_raw=df_us_macro_raw,
        df_us_benchmark_yields_raw=df_us_benchmark_yields_raw,
        df_us_tic_raw=df_us_tic_raw,
        df_global_yields_raw=df_global_yields_raw,
        global_macro_panels_raw=global_macro_panels_raw,
        raw_study_config=raw_study_config,
        study_metadata=study_metadata
    )
    
    # 4. Access results
    tables = artifacts["Tables"]
    print(tables["Combination_RMSFE_H1"].head())
```

## Output Structure

The pipeline returns a dictionary containing:
-   **`Tables`**: A dictionary of result DataFrames (e.g., `DNS_RMSFE`, `FADNS_Best_k`, `Combination_RMSFE_H1`).
-   **`Figures`**: A dictionary of DataFrames formatted for plotting (e.g., `Weight_Dynamics_Data`, `Global_SHAP_Data`).
-   **`Audit`**: A dictionary containing the frozen configuration and its cryptographic hash.

## Project Structure

```
forecasting_the_US_treasury_yield_curve/
│
├── forecasting_the_US_treasury_yield_curve_draft.ipynb   # Main implementation notebook
├── config.yaml                                           # Master configuration file
├── requirements.txt                                      # Python package dependencies
│
├── LICENSE                                               # MIT Project License File
└── README.md                                             # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Global Settings:** `start_date_us`, `forecast_horizons`, `us_zero_maturities`.
-   **Model Architectures:** `dns_fadns_window_w`, `rf_training_window_W`, `pca_k_grid`.
-   **Forecast Combination:** `fc_weight_window_W`, `fc_min_observations`, `alpha` (ES level), `eta` (robustness).
-   **Preprocessing:** `cusum_model_specification`, `rbf_kernel_setting`.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Machine Learning Models:** Integrating Gradient Boosting Machines (XGBoost, LightGBM) or Neural Networks (LSTM, Transformer) into the ensemble.
-   **Dynamic Factor Selection:** Implementing time-varying factor selection for the FADNS model.
-   **Real-Time Forecasting:** Connecting the pipeline to live data feeds for real-time yield curve prediction.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{liu2026forecasting,
  title={Forecasting the U.S. Treasury Yield Curve: A Distributionally Robust Machine Learning Approach},
  author={Liu, Jinjun and Cheng, Ming-Yen},
  journal={arXiv preprint arXiv:2601.04608},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). Forecasting U.S. Treasury Yields Replication Pipeline: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/forecasting_the_US_treasury_yield_curve
```

## Acknowledgments

-   Credit to **Jinjun Liu and Ming-Yen Cheng** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, Scikit-Learn, SciPy, Statsmodels, SHAP, and Ruptures**.

--

*This README was generated based on the structure and content of the `forecasting_the_US_treasury_yield_curve_draft.ipynb` notebook and follows best practices for research software documentation.*
