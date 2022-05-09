# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:12:35 2022

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
        anom, error, reconst, sig, latent_space = model.predict(True,
                                                              df_X,
                                                              only_predict=False,
                                                              load_alpha=True
                                                        )
    
    anom.to_csv(settings['model_name']+'_anomaly_predictions.csv')
    error.to_csv(settings['model_name']+'_abs_error.csv')
    reconst.to_csv(settings['model_name']+'_reconstructions.csv')
    sig.to_csv(settings['model_name']+'_sigma.csv')
    latent_space.to_csv(settings['model_name']+'_latent_space.csv')




    
