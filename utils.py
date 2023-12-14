# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:34:43 2022

@author: gastong@fing.edu.uy
"""


import pandas as pd
import numpy as np
import pickle


def set_index(dataRaw):
    col_time = dataRaw.columns[0]
    dataRaw[col_time] = pd.to_datetime(dataRaw[col_time])
    dataRaw = dataRaw.set_index(col_time)
    return dataRaw


def preprocessing(dataRaw=None, flag_scaler=True, scaler=None, scaler_name=None,
                  outliers=False, max_std=7, instance='fit'):
    
    X = dataRaw.copy()
    
    if outliers:
        z_scores = (X-X.mean())/X.std()
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores > max_std)
        X = X.mask(filtered_entries)

    X.fillna(method='ffill', inplace=True) 
    X.fillna(method='bfill', inplace=True)

    if flag_scaler:
        if instance == 'fit':
            X = scaler.fit_transform(X)
            pickle.dump(scaler, open(scaler_name + '_scaler.pkl','wb'))
        elif instance == 'transform':
            scaler = pickle.load(open(scaler_name + '_scaler.pkl','rb'))
            X = scaler.transform(X)
        elif instance == 'inverse':
            scaler = pickle.load(open(scaler_name + '_scaler.pkl','rb'))
            X = scaler.inverse_transform(X)
    
    X = pd.DataFrame(X, index=dataRaw.index, columns=dataRaw.columns)
    return X     

def MTS2UTS(ds=None, T=32, seed=42):
    N, C = ds.shape
    columns = ds.columns
    index = ds.index
    
    samples_index = []
    samples_values = []
    samples_class = []
    for c in range(C):
        serie_index = index
        serie_values = ds.iloc[:,c].values
        serie_name = columns[c]
        samples_aux_index = [serie_index[i: i + T] for i in range(0, N - T+1)]
        samples_aux_values = [serie_values[i: i + T] for i in range(0, N - T+1)]  
        samples_aux_class = [serie_name for s in range(len(samples_aux_values))]
        samples_index += samples_aux_index
        samples_values += samples_aux_values
        samples_class += samples_aux_class

    return samples_values, samples_index, samples_class    


def UTS2MTS(ds_val, ds_ix, ds_class):
    ds_val = np.array(ds_val)
    ds_ix = np.array(ds_ix)
    ds_class = np.array(ds_class)
    columns = ds_class[np.where(np.roll(ds_class,1)!=ds_class)[0]]
    if len(columns) == 0:
        columns = [ds_class[0]]
        index = ds_ix[:, -1]  
    else:
        index = ds_ix[:, -1][ds_class==columns[0]]
    df = pd.DataFrame(index=index, columns=columns)
    for col in columns:
        if ds_val.ndim > 1:
            df[col] = ds_val[:, -1][ds_class==col]
        else:
            df[col] = ds_val[ds_class==col]
    return df