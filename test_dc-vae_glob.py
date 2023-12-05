# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:12:35 2022

@author: gastong@fing.edu.uy
"""


import sys
import pandas as pd
import json
from dcvae_glob import DCVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':

    data_path = sys.argv[1]
    settings_path = sys.argv[2]
    
    # Data
    data  = pd.read_csv(data_path)

    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    data = set_index(data)
    data = preprocessing(data, flag_scaler=False, outliers=False)
    print(data.head())

    #Read dataframe with the quantiles, calculated in the training set
    quantiles = pd.read_csv(settings['model_name']+'_quantiles.csv', header=None, index_col=0, squeeze=True)

    data = data/quantiles

    # Model initialization
    model = DCVAE(T=settings['T'],
                  batch_size=settings['batch_size'])   
    
    # Alpha definition
    with tf.device('/gpu:0'):
        anom, reconst, sig, latent_space, sam_ix, sam_class = model.predict(data,
                                                              model_name=settings['model_name'],
                                                              only_predict=False,
                                                              load_alpha=False,
                                                              alpha_set=np.ones(12),
                                                        )
    
    anom.to_csv(settings['model_name']+'_anomaly_predictions.csv')
    reconst.to_csv(settings['model_name']+'_reconstructions.csv')
    sig.to_csv(settings['model_name']+'_sigma.csv')
