import pandas as pd
import numpy as np

def calculate_data_shape(dataframe):
    return dataframe.shape

def take_columns(dataframe):
    return dataframe.columns

def calculate_target_ratio(dataframe, target_name):
    return round(dataframe[target_name].mean(), 2)

def calculate_data_dtypes(x):
    return [x.dtypes[x.dtypes != 'object'].count(), x.dtypes[x.dtypes == 'object'].count()]

def calculate_cheap_apartment(x):
    return x[x['price_doc'] <= 1000000]['price_doc'].count()

def calculate_squad_in_cheap_apartment(x):
    return x[x['price_doc'] <= 1000000]['full_sq'].mean()

def calculate_mean_price_in_new_housing(x):
    return round(x[(x['build_year'] >= 2010) & (x['num_room'] == 3)]['price_doc'].mean())

def calculate_mean_squared_by_num_rooms(x):
    return round(x.groupby(['num_room'])['full_sq'].mean(), 2)    

def calculate_squared_stats_by_material(x):
    return round(x.groupby(['material'])['full_sq'].aggregate(['min', 'max']), 2)  

def calculate_crosstab(x):
    return round(x.groupby(['sub_area', 'product_type'])['price_doc'].aggregate(['mean']), 2)  