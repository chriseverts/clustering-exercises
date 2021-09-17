import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 

def plot_variable_pairs(df):
    '''accepts a dataframe as input
    split into train, validate, and test
    plots all of the pairwise relationships along with the regression line for each pair.'''
    plot = sns.pairplot(df, corner = True, kind = 'reg', plot_kws={'line_kws':{'color':'red'}})
    return plot

def months_to_years(df):
    """
    Takes in the telco df and returns the df with new 
    categorical feature 'tenure_years'
    """
    df['tenure_years'] = round(df.tenure // 12)
    df['tenure_years'] = df.tenure_years.astype('object')
    return df

def plot_categorical_and_continuous_vars(df, cat_vars, quant_vars):
    '''takes in a dataframe as input, with a discrete, and continuous variable and returns 
    and barplot, violin plot, boxplot'''
    sns.barplot(data=df, y=quant_vars, x=cat_vars)
    plt.show()
    sns.violinplot(data=df, y=quant_vars, x=cat_vars)
    plt.show()
    sns.boxplot(data=df, y=quant_vars, x=cat_vars)