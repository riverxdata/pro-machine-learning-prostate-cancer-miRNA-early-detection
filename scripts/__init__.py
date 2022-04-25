# Run parralell
import pyblaze.multiprocessing as xmp
# Basic process
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Feature Selection
import itertools
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Can get the functions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
# Can not get the functions
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Parameters optimization
from sklearn.model_selection import GridSearchCV
# Performance
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# Ignore wanring
import warnings
warnings.filterwarnings("ignore")
