# -*- coding: utf-8 -*-
"""
Created on Mon Aug  22 2022

@author: gastong@fing.edu.uy
"""


import sys
import pandas as pd
import json
from dc_vae import DCVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt
import tensorflow as tf


if __name__ == '__main__':

    data_path = sys.argv[1]
    settings_path = sys.argv[2]
    
    # Data
    data  = pd.read_csv(data_path)

    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    df_X = set_index(data)
    df_X = preprocessing(df_X, settings['scale'], None, settings['model_name'],
                         settings['wo_outliers'], settings['max_std'], 'transform')
    
    # Model initialization
    model = DCVAE(name=settings['model_name'],
                  T=settings['T'],
                  batch_size=settings['batch_size'])   
    
    # Alpha definition
    with tf.device('/cpu:0'):
        loss, reconstruction, kl = model.evaluate(True, df_X)
    
    print('Loss: ', loss)
    print('Reconstruction: ', reconstruction)
    print('KL: ', kl)



    
