
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 2022

@author: gastong@fing.edu.uy
"""

import sys
import pandas as pd
import numpy as np
import json
from dc_vae_gen import DCVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import optuna



def objective(trial):

    settings_path = sys.argv[1]
    
    # Data
    print('Reading the data...')
    #path = "../../Datasets/TELCO/"
    path = "/home/gastong/Documentos/TELCO/v0/" #Rosaluna

    filenames_train = ["TELCO_data_2021_01.zip", "TELCO_data_2021_02.zip", "TELCO_data_2021_03.zip"]
    files = [path+ i for i in filenames_train]
    data_train = pd.concat(map(pd.read_csv, files))
    data_train = set_index(data_train)
    data_train = preprocessing(data_train, flag_scaler=False, outliers=True)

    filenames_val = ["TELCO_data_2021_04.zip"]
    files = [path+ i for i in filenames_val]
    data_val = pd.concat(map(pd.read_csv, files))
    data_val = set_index(data_val)
    data_val = preprocessing(data_val, flag_scaler=False, outliers=True)

    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    print('Preprocessing the data...')
    data_train = data_train/data_train.quantile(0.98)
    data_val = data_val/data_train.quantile(0.98)


    # Model initialization
    model = DCVAE(
        trial.suggest_categorical("length_seq", [32, 128, 512]),
        settings['M'],
        [trial.suggest_categorical("units", [16, 32, 64])
         for i in range(int(np.log2(trial.suggest_categorical("length_seq", [32, 128, 512]))))],
        [int(2**i) for i in range(int(np.log2(trial.suggest_categorical("length_seq", [32, 128, 512]))))], 
        settings['kernel'],
        settings['strs'],
        trial.suggest_categorical("batch_size", [32, 64]),
        trial.suggest_categorical("J", [8, 4, 2]),
        settings['epochs'],
        trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        settings['lr_decay'],
        settings['decay_rate'],
        settings['decay_step'],
        settings['model_name'],
        summary = settings['summary']
        )   

    # Train
    model.fit(data_train, settings['val_percent'], settings['seed'])

    # Evaluate
    loss, reconstruction, kl = model.evaluate(False, data_val)
    
    
    return loss

if __name__ == '__main__':

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
