# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:08:16 2022

@author: gastong@fing.edu.uy
"""

import sys
import pandas as pd
import json
from dc_vae import DCVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt

if __name__ == '__main__':

    data_path = sys.argv[1]
    settings_path = sys.argv[2]
    
    # Data
    data_train  = pd.read_csv(data_path)
    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    df_X = set_index(data_train)
    df_X = preprocessing(df_X, settings['scale'], None, settings['model_name'],
                         settings['wo_outliers'], settings['max_std'], 'transform')
    
    # Model initialization
    model = DCVAE(
        settings['T'],
        settings['M'],
        settings['cnn_units'],
        settings['dil_rate'],
        settings['kernel'],
        settings['strs'],
        settings['batch_size'],
        settings['J'],
        settings['epochs'],
        settings['lr'],
        settings['decay_rate'],
        settings['decay_step'],
        settings['model_name'],
        summary = settings['summary']
        )   
    
    # Train
    model.fit(df_X, settings['val_percent'], settings['seed'])
    
    #Plot loss curves
    plt.plot(model.history_.history["loss"], label="Training Loss")
    plt.plot(model.history_.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig(settings['model_name'] + '_loss.jpg')