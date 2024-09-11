import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import utils.processing as pr
import utils.s3_utils as s3
from sklearn.preprocessing import PolynomialFeatures
from folium import plugins
import config.paths as path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LassoCV, LinearRegression, Lasso, RidgeCV, Ridge, LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
import random

def xgboost_model(x,y,eval_method = 'f1_weighted', params=False):

    # Initialize a new XGBClassifier with the best parameters found
    if params == False:
        regxGB = xgb.XGBClassifier(n_estimators=5000,
                                    max_depth=20,
                                    learning_rate= .002,
                                    colsample_bytree =.4,
                                    min_split_loss= 0,
                                    min_child_weight= 0,
                                    subsample=.4,
                                    reg_lambda=1.5,
                                    reg_alpha=1.)
    else:
        clf=params
        regxGB = xgb.XGBClassifier(n_estimators=clf['n_estimators'],
                            max_depth=clf['max_depth'],
                            learning_rate=clf['learning_rate'],
                            colsample_bytree=clf['colsample_bytree'],
                            min_split_loss=clf['min_split_loss'],
                            min_child_weight=clf['min_child_weight'],
                            subsample=clf['subsample'],
                            reg_lambda=clf['reg_lambda'],
                            reg_alpha=clf['reg_alpha'])


    # Fit the model on the data
    regxGB.fit(x, y)

    #get feature importances and sort high to low
    importances = pd.DataFrame(dict(zip(list(x.columns),regxGB.feature_importances_)), 
                           index = [1]).T
    importances.columns = ['feature_score']
    importances = importances.sort_values('feature_score',ascending=False)

    return regxGB, importances


def logreg_model(x_scaled, y, x):

    #Create model with best regularization parameter:
    lasso = LogisticRegression(max_iter = 10000,solver = "saga", penalty = "l1", random_state = 123)
    lasso.fit(x_scaled, y)

    # get the coefficients
    #get highest weights
    ta_orig= pd.DataFrame(zip(lasso.coef_[0], x.columns), columns = ['coeff', 'cols']).sort_values('coeff', ascending = False)
    label_dict = {
        'fg-' : 'windSpeed-',
        'tg-' : 'meanTemp-',
        'tn-' : 'minTemp-',
        'tx-' : 'maxTemp-',
        'rr-' : 'Precip-',
        'hu-' : 'relHumidity-',
        'qq-' : 'meanRadiation-',
        'cat_': ''
    }

    ta_orig['coeff-abs-value'] = ta_orig['coeff'].abs()
    importances = ta_orig


    return lasso, importances