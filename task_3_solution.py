import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

def split_data_into_two_samples(x):
    return train_test_split(x, test_size=0.3, random_state=42) 

def prepare_data(x):
    target_vector = x['price_doc']
    objects = x.select_dtypes(include=['object']).columns
    filtered = x.drop(columns=objects).drop(columns=['id']).drop(columns=['price_doc']).dropna(axis=1)
    return [filtered, target_vector]  

def scale_data(x, transformer):
    return transformer.fit_transform(x)

def prepare_data_for_model(x, transformer):
    x_train, y_train = prepare_data(x)
    scaled_x = scale_data(x_train, transformer)
    x_train_scaled = pd.DataFrame(scaled_x, columns=x_train.columns)
    return [x_train_scaled, y_train]

def fit_first_linear_model(x_train, y_train):
    x_train_s = scale_data(x_train, StandardScaler())
    clr = LinearRegression()
    clr.fit(x_train_s, y_train)
    return clr

def fit_first_linear_model2(x_train, y_train):
    x_train_s = scale_data(x_train, MinMaxScaler())
    clr = LinearRegression()
    clr.fit(x_train_s, y_train)
    return clr

def evaluate_model(linreg, x_pred, y_true):
    y_pred = linreg.predict(x_pred)
    round_num = 2
    mse = round(mean_squared_error(y_true, y_pred), round_num)
    mae = round(mean_absolute_error(y_true, y_pred), round_num)
    r2 = round(r2_score(y_true, y_pred), round_num)
    return [mse, mae, r2]

def calculate_model_weights(model, features):
    df = pd.DataFrame({'features': features, 'weights': model.coef_})
    df.sort_values(by=['weights'])
    return df