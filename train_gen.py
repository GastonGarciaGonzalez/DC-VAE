# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:08:16 2022

@author: gastong@fing.edu.uy
"""
from comet_ml import Experiment
import sys
import pandas as pd
import json
from dc_vae_gen import DCVAE
from utils import set_index, preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


experiment = Experiment(
    api_key = "VZhK7C4klolOVuvJAQ1OrekYt",
    project_name = "dc-vae-comparation",
    workspace="gastong",
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,

)

experiment.set_name('multivariado_nocond_gen')

if __name__ == '__main__':

    #data_path = sys.argv[1]
    settings_path = sys.argv[1]

    # Data
    print('Reading the data...')
    #path = "../../Datasets/TELCO/"
    path = "/home/gastong/Documentos/TELCO/v0/" #Rosaluna
    filenames = ["TELCO_data_2022_01.zip", "TELCO_data_2022_02.zip", "TELCO_data_2022_03.zip"]
    data = pd.DataFrame()
    for i in range(len(filenames)):
        data = pd.concat([data, pd.read_csv(path+filenames[i])])
    # Parameters
    settings = json.load(open(settings_path, 'r'))

    # Preprocess
    print('Preprocessing the data...')
    sc = StandardScaler()
    df_X = set_index(data)
    df_X = preprocessing(df_X, settings['scale'], sc, settings['model_name'],
                            settings['wo_outliers'], settings['max_std'], 'fit')

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
        settings['lr_decay'],
        settings['decay_rate'],
        settings['decay_step'],
        settings['model_name'],
        summary = settings['summary']
        )   

    # Train
    with experiment.train():
        model.fit(df_X, settings['val_percent'], settings['seed'])

    #Plot loss curves
    plt.plot(model.history_.history["loss"], label="Training Loss")
    plt.plot(model.history_.history["val_loss"], label="Validation Loss")
    plt.plot(model.history_.history["reconst"], label="Training Reconstruction")
    plt.plot(model.history_.history["val_reconst"], label="Validation Reconstruction")
    plt.plot(model.history_.history["kl"], label="Training KL")
    plt.plot(model.history_.history["val_kl"], label="Validation KL")

    plt.legend()
    plt.savefig(settings['model_name'] + '_loss.jpg')

    # Report multiple hyperparameters using a dictionary:
    hyper_params = settings
    experiment.log_parameters(hyper_params)
    experiment.log_dataset_hash(df_X) #creates and logs a hash of your data