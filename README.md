# TiO<sub>2</sub> Nanotoxicity Model

## Data
TiO<sub>2</sub> nanotoxicity data based from [Mikolajczyk, et al](https://pubs.rsc.org/en/content/articlelanding/2018/en/c8en00085a).

## Models
Total of 10 trained models are stored under `models`.

Models based on features: `Empirical`, `DFT`
Models based on ML algorithn: `MLR` (multiple linear regression), `KRR` (kernel ridge regression), `SVR` (support vector regression), `GPR` (Gaussian process regression), `RFR` (random forest regression)


## Script
`test_models.py`: This script runs the trained models for TiO2 nanotoxicity prediction, and tests each model performance based on RMSE, MAE and R2. It also checks the applicability domain (AD) of each predicted sample.
