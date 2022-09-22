import os
import time
import caffeine

import numpy as np
import pandas as pd 
import pickle

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor

from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize

import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

#--------------------------------------------------------------
# Main function
#--------------------------------------------------------------
if __name__ == "__main__":
  print("-------------------------------------------------------")
  print("     Nanotoxicity Prediction via Regression Models     ")
  print("-------------------------------------------------------\n")
  
  #--------------------------------------------------------------
  # Initialize settings
  warnings.simplefilter("ignore", category=ConvergenceWarning)

  rnd = 2021
  np.random.seed(rnd)

  SC, scoring = 'rmse', 'neg_root_mean_squared_error'
  
  #--------------------------------------------------------------
  # Read data from file fname
  fname = 'tio2_data.csv'
  fpath = os.path.join(os.getcwd(), 'data', fname)
  print("Loading data set [%s]..." % fname)

  df = pd.read_csv(fpath) 

  # data = data.apply(pd.to_numeric, errors='ignore')
  df['Sample'] = df['Sample'].astype('str')
  df.loc[:, df.columns != 'Sample'] = df.drop('Sample', axis=1).astype(float)

  # Test samples index
  ti = [3, 8, 11, 15, 18, 22, 26, 30]

  #--------------------------------------------------------------
  # Prepare data
  #--------------------------------------------------------------
  # Remove non-essential columns
  df = df.drop('ID', axis=1)
  df = df.drop('Sample', axis=1)
  df = df.drop('Vis', axis=1)
  df = df.drop('UVVis', axis=1)

  # Designate feature variables
  feat_all = list(df.columns)
  feat_all.remove('Toxicity')

  # Prepare training and test sets
  y_raw = df['Toxicity']
  x_raw = df[feat_all]

  x_raw = x_raw.to_numpy()
  y_raw = y_raw.to_numpy()
  y_raw = y_raw.reshape(-1, 1)

  N, M = x_raw.shape[0], x_raw.shape[1]
  print("- Samples: %d" % N)
  print("- Features:", feat_all)


  #--------------------------------------------------------------
  # Test models
  #--------------------------------------------------------------
  # Separate training and test sets
  test_idx = np.full(N, False, dtype=bool)
  test_idx[ti] = True

  x_train, y_train = x_raw[~test_idx, :], y_raw[~test_idx, :]
  x_test, y_test   = x_raw[test_idx, :],  y_raw[test_idx, :]

  # Load pickled models
  mod_names = ['Empirical_MLR', 
               'Empirical_KRR',
               'Empirical_SVR',
               'Empirical_GPR',
               'Empirical_RFR',
               'DFT_MLR', 
               'DFT_KRR',
               'DFT_SVR',
               'DFT_GPR',
               'DFT_RFR',]
  selected_features = [['Elen', 'Ion_E'],
                       ['Cov_rad'],
                       ['SSE'],
                       ['Mass_g', 'Elea'],
                       ['SSE'],
                       ['Abs_hard'],
                       ['Abs_hard'],
                       ['Abs_elen'],
                       ['Abs_hard'],
                       ['Abs_elen', 'Ea']]

  for mn, sf in zip(mod_names, selected_features):
    print("\n----------------------------------")
    print("- Model :", mn)
    print("----------------------------------")
    
    # Create copies of mutable objects to avoid overwriting
    x_train_, y_train_ = np.copy(x_train), np.copy(y_train)
    x_test_, y_test_ = np.copy(x_test), np.copy(y_test)

    # Select feature/s
    fi = [i for i in range(len(feat_all)) if feat_all[i] in sf]
    x_train_cv = x_train_[:,fi]
    x_test_cv  = x_test_[:,fi]

    sc_x, sc_y = StandardScaler(), StandardScaler()
    x_train_cv = sc_x.fit_transform(x_train_cv)
    y_train_cv = sc_y.fit_transform(y_train_)

    x_test_cv = sc_x.transform(x_test_cv)
    y_test_cv = sc_y.transform(y_test_)


    # Load model
    file_name = "%s.mod" % mn
    file_path = os.path.join(os.getcwd(), 'models', file_name)
    model_ = pickle.load(open(file_path, 'rb'))
    
    # Test model
    y_pred = model_.predict(x_test_cv)

    y_pred_raw = np.squeeze(sc_y.inverse_transform(y_pred))
    y_test_raw = np.squeeze(sc_y.inverse_transform(y_test_cv))

    # Compute applicability domain
    ad_X = x_train_cv
    ad_q = x_test_cv

    ad_samp = np.linalg.inv(np.dot(ad_X.T, ad_X))
    h = [q.T.dot(ad_samp).dot(q) for q in ad_q]

    ad_threshold = 3 * len(sf)/x_train_cv.shape[0]
    y_ad = [d<=ad_threshold for d in h]

    print("\nTest predictions within AD: %d/%d" % (sum(y_ad), x_test_cv.shape[0]))
    
    # Compute model performance
    r2 = model_.score(x_test_cv, y_test_cv)
    rmse = mean_squared_error(y_test_raw, y_pred_raw, squared=False)
    mae = mean_absolute_error(y_test_raw, y_pred_raw)

    print("\n[Features]", end =" ") 
    print(*sf, sep = ", ") 
    print("\n[Test Score] RMSE: %.6f | MAE: %.6f | R2: %.6f\n" % (rmse, mae, r2))
