################################################
#Python default module loads for various stages:
################################################

#####
#EDA:
#####

import os
import sys

import pandas as pd
import numpy as np
import psycopg2

import seaborn as sns
import matplotlib.pyplot as plt

from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
pd.options.mode.chained_assignment = None  # default='warn'

from pylab import rcParams
rcParams['figure.figsize'] = 12, 5

########
#Models:
########

import pandas as pd
import numpy as np
import psycopg2

import xgoost as xgb
from sklearn.linear_models import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib












