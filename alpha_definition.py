# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:55:00 2022

@author: gastong@fing.edu.uy
"""

import sys
import pandas as pd
import json
from dc_vae import DCVAE
from utils import set_index, preprocessing
import tensorflow as tf

if __name__ == '__main__':

    data_path = sys.argv[1]
    labels_path = sys.argv[2]
    settings_path = sys.argv[3]
    
    # Data
    data  = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)
    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    df_X = set_index(data)
    df_X = preprocessing(df_X, settings['scale'], None, settings['model_name'],
                         settings['wo_outliers'], settings['max_std'], 'transform')
    df_y = set_index(labels)
    
    # Model initialization
    model = DCVAE(name=settings['model_name'],
                  T=settings['T'],
                  batch_size=settings['batch_size'])   
    
    # Alpha definition
    with tf.device('/cpu:0'):
        model.alpha_selection(True, df_X, df_y, settings['custom_metrics'])
    
    # Results
    print('Alpha up: ', model.alpha_up)
    print('Alpha down: ', model.alpha_down) 
    print('max_F1: ', model.f1_val) 