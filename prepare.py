import sklearn.preprocessing

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

def scale_data(train, validate, test):
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    train_scaled = pd.DataFrame(train_scaled, columns = train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns = train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns = train.columns)
    
    return train_scaled, validate_scaled, test_scaled