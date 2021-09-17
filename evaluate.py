import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics

def plot_residuals(y,yhat):
    '''takes in actual column and predicted column, and creates a residual plot'''
    residuals = y - yhat
    plt.scatter(y, residuals)
    plt.axhline(y=0, color='black')
    plt.show()

def regression_errors(y, yhat):

    residuals = yhat - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    print(f'SSE is {SSE}')
    
    ESS = sum((yhat - y.mean())**2)
    print(f'ESS is {ESS}')
    
    TSS = ESS + SSE
    print(f'TSS is {TSS}')
    
    MSE = sklearn.metrics.mean_squared_error(y,yhat)
    print(f'MSE is {MSE}')
    
    RMSE = math.sqrt(sklearn.metrics.mean_squared_error(y,yhat))
    print(f'RMSE is {RMSE}')

def baseline_mean_errors(y):
    import sklearn.metrics
    import math
    baseline = y.mean()
    residuals = baseline - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    print(f'SSE baseline is {SSE}')
    
    MSE = SSE/len(y)
    print(f'MSE baseline is {MSE}')
    
    RMSE = math.sqrt(MSE)
    print(f'RMSE baseline {RMSE}')


def better_than_baseline(y,yhat):
    baseline = y.mean()
    residuals_baseline = baseline - y
    residuals_squared_baseline = sum(residuals_baseline**2)
    SSE_baseline = residuals_squared_baseline
    
    MSE_baseline = SSE_baseline/len(y)
    
    RMSE_baseline = math.sqrt(MSE_baseline)
    
    residuals = yhat - y
    residuals_squared = sum(residuals**2)
    SSE = residuals_squared
    
    MSE = sklearn.metrics.mean_squared_error(y,yhat)
    
    RMSE = math.sqrt(sklearn.metrics.mean_squared_error(y,yhat))
    
    if RMSE < RMSE_baseline:
        return True
    else: 
        return False


def model_significance(ols_model):
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    print(f'variance explained in our model is {round(r2,2)}')
    if f_pval <.05:
        return print(f"our p-value for our model's significance is {f_pval}, so it is significantly better than the baseline")
    else:
        return print(f"our p-value for our model's significance is {f_pval}, so it is NOT significantly better than the baseline")
