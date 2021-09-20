import pandas as pd 
import numpy as np
import os 
from env import username, host, password
from sklearn.model_selection import train_test_split


    # Acquiring telco_churn data
def get_connection(db, username=username, host=host, password=password):
    '''
    Creates a connection URL
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def new_telco_charge_data():
    '''
    Returns telco_charge into a dataframe
    '''
    sql_query = '''select customer_id, monthly_charges, tenure, total_charges from customers
    join internet_service_types using(internet_service_type_id)
    join contract_types using(contract_type_id)
    join payment_types using(payment_type_id)
    where contract_type_id = 3'''
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    return df 

def get_telco_charge_data():
    '''get connection, returns telco_charge into a dataframe and creates a csv for us'''
    if os.path.isfile('telco_charge.csv'):
        df = pd.read_csv('telco_charge.csv', index_col=0)
    else:
        df = new_telco_charge_data()
        df.to_csv('telco_charge.csv')
    return df

def clean_telco(df):
    '''cleans our data'''
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value=df.total_charges.mean()).astype('float64')
    df = df.set_index("customer_id")
    return df 


def split_telco(df):
    '''
    Takes in a cleaned df of telco data and splits the data appropriatly into train, validate, and test.
    '''
    
    train_val, test = train_test_split(df, train_size =  0.8, random_state = 123)
    train, validate = train_test_split(train_val, train_size =  0.7, random_state = 123)
    return train, validate, test

def wrangle_telco():
    '''acquire and our dataframe, returns a df'''
    df = clean_telco(get_telco_charge_data())
    return df


def wrangle_split_telco():
    '''acquire, clean, split our dataframe'''
    df = clean_telco(get_telco_charge_data())
    return split_telco(df)

#Zillow

def new_zillow():
    sql_query ='''select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017
 	join propertylandusetype using(propertylandusetypeid)
 	where propertylandusetypeid = 261'''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df 

def get_zillow_data():
    '''get connection, returns Zillow into a dataframe and creates a csv for us'''
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow()
        df.to_csv('zillow.csv')
    return df

def wrangle_zillow():
    '''
    Read zillow csv file into a pandas DataFrame,
    only returns desired columns and single family residential properties,
    drop any rows with Null values, drop duplicates,
    return cleaned zillow DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv('zillow.csv')
    
    # Drop nulls
    df = df.dropna()
    
    # Drop duplicates
    df = df.drop_duplicates()
    # drop unamed column
    df = df.drop(columns=['Unnamed: 0'])
    return df

def split_zillow(df):
    '''
    Takes in a cleaned df of zillow data and splits the data appropriatly into train, validate, and test.
    '''
    
    train_val, test = train_test_split(df, train_size =  0.8, random_state = 123)
    train, validate = train_test_split(train_val, train_size =  0.7, random_state = 123)
    return train, validate, test

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def get_mall_customers():
    
    file_name = 'mall_customers.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        sql_query =  '''select * from customers'''
    df = pd.read_sql(sql_query, get_connection('mall_customers'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df


def split_mall_customers(df):
    '''
    Takes in a cleaned df of mall_customers data and splits the data appropriatly into train, validate, and test.
    '''
    
    train_val, test = train_test_split(df, train_size =  0.8, random_state = 123)
    train, validate = train_test_split(train_val, train_size =  0.7, random_state = 123)
    return train, validate, test

def impute(df, my_strategy, column_list):
    ''' 
    This function takes in a df, strategy, and column list and
    returns df with listed columns imputed using imputing stratagy
    '''
    # build imputer    
    imputer = SimpleImputer(strategy=my_strategy)  
    # fit/transform selected columns
    df[column_list] = imputer.fit_transform(df[column_list]) 

    return df