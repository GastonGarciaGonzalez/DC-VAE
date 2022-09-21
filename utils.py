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